import functools
import os

import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.device import DeviceInfo, get_device_index
from humming.tune.base import DeviceHeuristics
from humming.tune.raster import raster_group_m_for_config
from humming.tune.ppu_sm80 import PPUSm80Heuristics
from humming.tune.sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from humming.tune.sm75 import Sm75Heuristics
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm100 import Sm100Heuristics
from humming.tune.sm120 import Sm120Heuristics
from humming.tune.sm121 import Sm121Heuristics

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
    110: Sm100Heuristics,
    120: Sm120Heuristics,
    121: Sm121Heuristics,
}


ppu_heuristics_map: dict[int, type[DeviceHeuristics]] = {80: PPUSm80Heuristics}


def get_heuristics_class(device: int | torch.device | None = None) -> type[DeviceHeuristics]:
    info = DeviceInfo(device)
    sm_version = info.sm_version
    if info.is_ppu:
        return ppu_heuristics_map[80]
    if sm_version == 90:
        if "H20" in info.name and "H200" not in info.name:
            return Sm90H20Heuristics

    if sm_version in heuristics_map:
        return heuristics_map[sm_version]

    sm_version_base = sm_version // 10 * 10

    return heuristics_map[sm_version_base]


def _apply_m_major_input_scale(
    config: dict,
    use_m_major_input_scale: bool,
    layer_config: LayerConfig,
    gemm_type: GemmType,
) -> None:
    if not use_m_major_input_scale:
        return
    use_tma = config.get("use_tma", False)
    if use_tma and layer_config.input_scale_group_size > 0 and gemm_type != GemmType.INDEXED:
        config["use_tma_as"] = True


def _disable_indexed_input_scale_tma(config: dict, gemm_type: GemmType) -> None:
    if gemm_type == GemmType.INDEXED:
        config["use_tma_a"] = False
        config["use_tma_c"] = False
        config["use_tma_as"] = False
        config["use_tma_as2"] = False


def _apply_raster_group_m(config: dict, layer_config, gemm_type) -> None:
    if gemm_type != GemmType.DENSE:
        return
    if config.get("raster_group_m") is not None or "block_shape" not in config:
        return
    try:
        config["raster_group_m"] = raster_group_m_for_config(
            layer_config,
            config["block_shape"],
            config.get("multi_cast_size_a", 1),
        )
    except Exception:
        pass


def _is_rs_w4a8(
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
) -> bool:
    return (
        os.getenv("HUMMING_EXPERIMENTAL_RS_W4A8") == "1"
        and gemm_type == GemmType.GROUPED_CONTIGUOUS
        and layer_config.sm_version == 90
        and use_m_major_input_scale
        and layer_config.mma_type == MmaType.WGMMA
        and layer_config.a_dtype == dtypes.float8e4m3
        and layer_config.b_dtype == dtypes.float4e2m1
        and layer_config.as_dtype == dtypes.float32
        and layer_config.bs_dtype == dtypes.float8e8m0
        and layer_config.use_fused_e8m0_scale
        and layer_config.input_scale_group_size == 128
        and layer_config.weight_scale_group_size == 32
        and 1 <= layer_config.num_experts <= 256
    )


# Routed rows normalized to 32 experts. Measured for GLM-5.2's grouped MoE
# GEMMs on H200 with uneven routing. Tile-count discontinuities make the
# optimal M non-monotonic; the same per-expert load is used at other EP sizes.
_RS_W4A8_M_POLICY = (
    (1536, 64),
    (2560, 96),
    (3584, 128),
    (4608, 160),
    (5632, 176),
    (7680, 128),
    (8704, 144),
    (10240, 160),
    (13568, 144),
    (15360, 160),
    (17408, 176),
    (19456, 160),
    (22528, 176),
    (32768, 160),
)


def _rs_w4a8_use_shape_policy(layer_config: LayerConfig) -> bool:
    return (
        "H200" in torch.cuda.get_device_name()
        and (layer_config.shape_n, layer_config.shape_k)
        in ((4096, 6144), (6144, 2048))
    )


def _rs_w4a8_block_m(layer_config: LayerConfig, shape_m: int | None) -> int:
    # The shape policy was calibrated on H200. Keep the fixed M176 tile on H100
    # until that device is swept separately.
    if shape_m is None or not _rs_w4a8_use_shape_policy(layer_config):
        return 176
    normalized_m = (shape_m * 32 + layer_config.num_experts - 1) // layer_config.num_experts
    for upper, tile_m in _RS_W4A8_M_POLICY:
        if normalized_m <= upper:
            return tile_m
    return 176


def _apply_rs_w4a8(
    config: dict,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
    shape_m: int | None = None,
) -> None:
    if not _is_rs_w4a8(layer_config, use_m_major_input_scale, gemm_type):
        return

    block_m = _rs_w4a8_block_m(layer_config, shape_m)
    config.update(
        block_shape=(block_m, 128, 128),
        warp_shape=(block_m, 16, 128),
        num_stages=5 if block_m == 64 else 4,
        use_warp_spec=True,
        use_rs_w4a8=True,
        use_stream_k=False,
        use_packed_k_layout=False,
        raster_group_m=1,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )


@functools.lru_cache(maxsize=1024)
def _get_heuristics_config(
    layer_config: LayerConfig,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type: str | GemmType = "dense",
    device_index: int = 0,
):
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    heuristics_cls = get_heuristics_class(device=device_index)
    if isinstance(shape_m, int):
        config = heuristics_cls.get_config(
            layer_config=layer_config,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        _apply_m_major_input_scale(config, use_m_major_input_scale, layer_config, gemm_type)
        _disable_indexed_input_scale_tma(config, gemm_type)
        _apply_raster_group_m(config, layer_config, gemm_type)
        _apply_rs_w4a8(config, layer_config, use_m_major_input_scale, gemm_type, shape_m)
        return config
    else:
        configs = heuristics_cls.get_configs(
            layer_config=layer_config,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        auto_m = (
            _is_rs_w4a8(layer_config, use_m_major_input_scale, gemm_type)
            and _rs_w4a8_use_shape_policy(layer_config)
        )
        if not auto_m:
            for entry in configs:
                _apply_m_major_input_scale(
                    entry[2], use_m_major_input_scale, layer_config, gemm_type
                )
                _disable_indexed_input_scale_tma(entry[2], gemm_type)
                _apply_raster_group_m(entry[2], layer_config, gemm_type)
                _apply_rs_w4a8(entry[2], layer_config, use_m_major_input_scale, gemm_type)
            return configs

        boundaries = tuple(
            (upper * layer_config.num_experts + 31) // 32
            for upper, _ in _RS_W4A8_M_POLICY
        )
        tuned_configs = []
        for lower, upper, base_config in configs:
            cuts = [lower, *(x for x in boundaries if lower < x < upper), upper]
            for interval_lower, interval_upper in zip(cuts, cuts[1:]):
                config = dict(base_config)
                _apply_m_major_input_scale(config, use_m_major_input_scale, layer_config, gemm_type)
                _disable_indexed_input_scale_tma(config, gemm_type)
                _apply_raster_group_m(config, layer_config, gemm_type)
                _apply_rs_w4a8(
                    config, layer_config, use_m_major_input_scale, gemm_type, interval_upper
                )
                if (
                    tuned_configs
                    and tuned_configs[-1][1] == interval_lower
                    and tuned_configs[-1][2] == config
                ):
                    tuned_configs[-1][1] = interval_upper
                else:
                    tuned_configs.append([interval_lower, interval_upper, config])
        return tuned_configs


def get_heuristics_config(
    layer_config: LayerConfig | dict,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type: str | GemmType = "dense",
    device: int | torch.device | None = None,
):
    device_index = get_device_index(device)
    with torch.cuda.device(device_index):
        if isinstance(layer_config, dict):
            layer_config = LayerConfig(**layer_config)
        layer_config.check_device(device_index)
        return _get_heuristics_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale,
            gemm_type,
            device_index,
        )
