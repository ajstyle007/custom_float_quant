import numpy as np
import pytest

from custom_float_quant import CustomFloat


def test_block_scaled_preserves_shape():
    cf = CustomFloat.mxfp4_e2m1()
    arr = np.random.randn(4, 128)
    quantized, scales = cf.quantize_array_block_scaled(arr, block_size=32)
    assert quantized.shape == arr.shape


def test_block_scaled_no_overflow_with_local_outlier():
    """A block containing a huge outlier shouldn't overflow, and other
    blocks shouldn't be crushed by an outlier that isn't even in them."""
    cf = CustomFloat.mxfp4_e2m1()
    arr = np.concatenate([
        np.full(32, 100.0),  # block 0: outlier-dominated
        np.full(32, 0.3),    # block 1: small, uniform, no outlier
    ])
    quantized, scales = cf.quantize_array_block_scaled(arr, block_size=32)
    assert np.all(np.isfinite(quantized))
    assert len(scales) == 2
    # block 1 shouldn't be crushed to zero the way whole-tensor scaling would
    assert np.abs(quantized[32:]).mean() > 0


def test_power_of_two_scale_is_power_of_two():
    cf = CustomFloat.mxfp4_e2m1()
    arr = np.random.uniform(-50, 50, size=64)
    _, scales = cf.quantize_array_block_scaled(arr, block_size=32, power_of_two_scale=True)
    for s in scales.flatten():
        log2_s = np.log2(s)
        assert np.isclose(log2_s, round(log2_s)), f"scale {s} is not a power of two"


def test_block_scaling_beats_whole_tensor_scaling_on_outlier_data():
    """The actual point of the feature: block scaling should recover more
    accuracy than whole-tensor scaling when a tensor has a local outlier
    that would otherwise crush every other value's precision."""
    cf = CustomFloat.mxfp4_e2m1()

    rng = np.random.default_rng(0)
    arr = np.concatenate([
        [500.0],                          # one big outlier
        rng.uniform(-1.0, 1.0, size=255),  # everything else is small
    ])

    whole_tensor_quantized, _ = cf.quantize_array_scaled(arr)
    block_quantized, _ = cf.quantize_array_block_scaled(arr, block_size=32)

    # skip the outlier itself (index 0) when measuring error on the "bulk"
    bulk_true = arr[1:]
    whole_tensor_err = np.abs(bulk_true - whole_tensor_quantized[1:]).mean()
    block_err = np.abs(bulk_true - block_quantized[1:]).mean()

    assert block_err < whole_tensor_err