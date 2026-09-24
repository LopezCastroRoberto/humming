import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.testing import KernelTestCase, KernelTestRunner
from humming.tune import get_heuristics_config


@pytest.mark.parametrize("num_experts", [4, 8, 16, 32, 64, 128, 256, 33])
@pytest.mark.parametrize("shape_m", [17, 512])
@pytest.mark.parametrize("top_k", [1, 8])
def test_grouped_mxfp4_fp8_m_major_rs(shape_m, num_experts, top_k, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("RS-WGMMA W4A8 requires SM90")
    if top_k > num_experts:
        pytest.skip("top-k cannot exceed the local expert count")

    monkeypatch.setenv("HUMMING_EXPERIMENTAL_RS_W4A8", "1")
    case = KernelTestCase(
        name="grouped-mxfp4-fp8-m-major-rs",
        layer_config=LayerConfig(
            shape_n=1024,
            shape_k=1024,
            num_experts=num_experts,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=128,
            weight_scale_group_size=32,
        ),
        compute_config=ComputeConfig(
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
            use_m_major_input_scale=True,
        ),
        top_k=top_k,
        seed=2026,
        atol=0.5,
    )
    results = KernelTestRunner(case).run(shape_ms=[shape_m])
    assert len(results) == 1
    assert results[0].tuning_config.use_rs_w4a8
    assert results[0].tuning_config.block_shape == (176, 128, 128)


@pytest.mark.parametrize("shape_n, shape_k", [(4096, 6144), (6144, 2048)])
def test_glm52_grouped_w4a8_shape_policy(shape_n, shape_k, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("RS-WGMMA W4A8 requires SM90")

    monkeypatch.setenv("HUMMING_EXPERIMENTAL_RS_W4A8", "1")
    layer_config = LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=32,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=128,
        weight_scale_group_size=32,
    )
    configs = get_heuristics_config(
        layer_config,
        use_m_major_input_scale=True,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    assert configs[0][0] == 0
    assert configs[-1][1] == 1 << 30
    assert all(left[1] == right[0] for left, right in zip(configs, configs[1:]))
    expected = {
        1024: (64, 5),
        3072: (128, 4),
        4096: (160, 4),
        5120: (176, 4),
        7168: (128, 4),
        10240: (160, 4),
        12288: (144, 4),
        14336: (160, 4),
        16384: (176, 4),
        18432: (160, 4),
        20480: (176, 4),
        24576: (160, 4),
    }
    for routed_m, tile_and_stages in expected.items():
        config = next(c for lo, hi, c in configs if lo < routed_m <= hi)
        assert (config["block_shape"][0], config["num_stages"]) == tile_and_stages
        single = get_heuristics_config(
            layer_config,
            shape_m=routed_m,
            use_m_major_input_scale=True,
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
        )
        assert (single["block_shape"][0], single["num_stages"]) == tile_and_stages


@pytest.mark.parametrize("num_experts", [4, 8, 16, 32, 64, 128, 256])
def test_glm52_grouped_w4a8_policy_scales_with_experts(num_experts, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("RS-WGMMA W4A8 requires SM90")

    monkeypatch.setenv("HUMMING_EXPERIMENTAL_RS_W4A8", "1")
    layer_config = LayerConfig(
        shape_n=4096,
        shape_k=6144,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=128,
        weight_scale_group_size=32,
    )
    configs = get_heuristics_config(
        layer_config,
        use_m_major_input_scale=True,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )
    for normalized_m, expected_tile_m in (
        (1024, 64),
        (2048, 96),
        (4096, 160),
        (8192, 144),
        (16384, 176),
    ):
        routed_m = normalized_m * num_experts // 32
        config = next(c for lo, hi, c in configs if lo < routed_m <= hi)
        direct = get_heuristics_config(
            layer_config,
            shape_m=routed_m,
            use_m_major_input_scale=True,
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
        )
        assert config["block_shape"][0] == expected_tile_m
        assert direct["block_shape"][0] == expected_tile_m


@pytest.mark.parametrize(
    "shape_n,shape_k,input_m,top_k,expected_m,expected_stages",
    [
        (4096, 6144, 1, 8, 64, 5),
        (4096, 6144, 128, 8, 64, 5),
        (4096, 6144, 1536, 8, 144, 4),
        (4096, 6144, 2048, 8, 176, 4),
        (6144, 2048, 8, 1, 64, 5),
        (6144, 2048, 1024, 1, 64, 5),
        (6144, 2048, 12288, 1, 144, 4),
        (6144, 2048, 14336, 1, 160, 4),
    ],
)
def test_glm52_grouped_w4a8_shape_policy_accuracy(
    shape_n, shape_k, input_m, top_k, expected_m, expected_stages, monkeypatch
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("RS-WGMMA W4A8 requires SM90")

    monkeypatch.setenv("HUMMING_EXPERIMENTAL_RS_W4A8", "1")
    case = KernelTestCase(
        name="glm52-grouped-w4a8-autotuned",
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=32,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=128,
            weight_scale_group_size=32,
        ),
        compute_config=ComputeConfig(
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
            use_m_major_input_scale=True,
        ),
        top_k=top_k,
        seed=2026,
        rtol=0.1,
        atol=32.0,
    )
    result = KernelTestRunner(case).run(shape_ms=[input_m])[0]
    assert result.tuning_config.block_shape == (expected_m, 128, 128)
    assert result.tuning_config.num_stages == expected_stages
    cosine = torch.nn.functional.cosine_similarity(
        result.outputs.float().flatten(), result.outputs_ref.float().flatten(), dim=0
    )
    assert cosine >= 0.9999
