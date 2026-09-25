"""Hopper MXFP4 x FP8(GS128) grouped-prefill tuning policy."""

import os

import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType


# Routed rows normalized to 32 experts. Measured for GLM-5.2's grouped MoE
# GEMMs on H200 with uneven routing. Tile-count discontinuities make the
# optimal M non-monotonic; the same per-expert load is used at other EP sizes.
_M_TILE_POLICY = (
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


def _enabled(
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
) -> bool:
    # Preserve the existing experimental switch used by serving and benchmarks.
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


def _use_tuned_m_tiles(layer_config: LayerConfig) -> bool:
    return (
        "H200" in torch.cuda.get_device_name()
        and (layer_config.shape_n, layer_config.shape_k)
        in ((4096, 6144), (6144, 2048))
    )


def _block_m(layer_config: LayerConfig, shape_m: int) -> int:
    normalized_m = (shape_m * 32 + layer_config.num_experts - 1) // layer_config.num_experts
    for upper, tile_m in _M_TILE_POLICY:
        if normalized_m <= upper:
            return tile_m
    return 176


def _set_config(config: dict, block_m: int) -> None:
    config.update(
        block_shape=(block_m, 128, 128),
        warp_shape=(block_m, 16, 128),
        num_stages=5 if block_m == 64 else 4,
        use_warp_spec=True,
        use_flat_grouped_raster=True,
        use_shared_as_promotion=True,
        use_stream_k=False,
        use_packed_k_layout=False,
        raster_group_m=1,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )


def apply_w4a8_config(
    config: dict,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
    shape_m: int,
) -> None:
    if not _enabled(layer_config, use_m_major_input_scale, gemm_type):
        return
    # H100 retains the measured fixed M176 tile until its own sweep is done.
    block_m = _block_m(layer_config, shape_m) if _use_tuned_m_tiles(layer_config) else 176
    _set_config(config, block_m)


def specialize_w4a8_ranges(
    configs: list,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
) -> list:
    if not _enabled(layer_config, use_m_major_input_scale, gemm_type):
        return configs

    if not _use_tuned_m_tiles(layer_config):
        for _, _, config in configs:
            _set_config(config, 176)
        return configs

    boundaries = tuple(
        (upper * layer_config.num_experts + 31) // 32
        for upper, _ in _M_TILE_POLICY
    )
    tuned_configs = []
    for lower, upper, base_config in configs:
        cuts = [lower, *(x for x in boundaries if lower < x < upper), upper]
        for interval_lower, interval_upper in zip(cuts, cuts[1:]):
            config = dict(base_config)
            _set_config(config, _block_m(layer_config, interval_upper))
            if (
                tuned_configs
                and tuned_configs[-1][1] == interval_lower
                and tuned_configs[-1][2] == config
            ):
                tuned_configs[-1][1] = interval_upper
            else:
                tuned_configs.append([interval_lower, interval_upper, config])
    return tuned_configs
