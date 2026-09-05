import math
import random

import numpy as np
import pytest

from custom_float_quant import CustomFloat, PRESETS


def _assert_scalar_vector_match(cf, values):
    scalar_results = [cf.quantize(float(v)) for v in values]
    vector_results = cf.quantize_array(np.array(values, dtype=np.float64))

    for v, s, vec in zip(values, scalar_results, vector_results):
        if math.isnan(s) and math.isnan(vec):
            continue
        if s == 0.0 and vec == 0.0:
            assert math.copysign(1.0, s) == math.copysign(1.0, vec), (
                f"signed zero mismatch for value={v!r}"
            )
            continue
        assert s == vec, f"value={v!r} scalar={s!r} vector={vec!r}"


def test_vectorized_matches_scalar_specials(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    specials = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    _assert_scalar_vector_match(cf, specials)


def test_vectorized_matches_scalar_random_normals(preset_name):
    random.seed(42)
    cf = CustomFloat.from_preset(preset_name)
    values = [random.uniform(-1e4, 1e4) for _ in range(300)]
    _assert_scalar_vector_match(cf, values)


def test_vectorized_matches_scalar_subnormal_range(preset_name):
    random.seed(42)
    cf = CustomFloat.from_preset(preset_name)
    values = [
        random.choice([1, -1]) * (2.0 ** e) * random.uniform(1.0, 2.0)
        for e in range(-160, -1, 4)
    ]
    _assert_scalar_vector_match(cf, values)


def test_vectorized_matches_scalar_overflow_range(preset_name):
    random.seed(42)
    cf = CustomFloat.from_preset(preset_name)
    values = [
        random.choice([1, -1]) * (2.0 ** e) * random.uniform(1.0, 2.0)
        for e in range(1, 300, 6)
    ]
    _assert_scalar_vector_match(cf, values)


def test_vectorized_matches_scalar_rne_ties(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    shift = 52 - cf.mantissa_bits
    if shift < 2:
        pytest.skip("mantissa_bits too wide for this format to construct a tie case")

    values = []
    for base_exp in range(-4, 5):
        base = 2.0 ** base_exp
        step = max(1, (1 << cf.mantissa_bits) // 8)
        for k in range(0, 1 << cf.mantissa_bits, step):
            tie_bit = 1 << (shift - 1)
            frac_bits = (k << shift) | tie_bit
            val = base * (1.0 + frac_bits / (1 << 52))
            values.append(val)
            values.append(-val)
    _assert_scalar_vector_match(cf, values)


def test_quantize_array_preserves_shape():
    cf = CustomFloat.mxfp8_e4m3()
    arr = np.random.randn(8, 16)
    out = cf.quantize_array(arr)
    assert out.shape == arr.shape
    assert out.dtype == np.float64


def test_quantize_array_scaled_prevents_overflow():
    """A tensor whose max value exceeds mxfp4's tiny range should not
    produce inf/nan once scaled."""
    cf = CustomFloat.mxfp4_e2m1()
    arr = np.array([100.0, -100.0, 0.5, -0.5, 0.0])
    quantized, scale = cf.quantize_array_scaled(arr)
    assert np.all(np.isfinite(quantized))
    assert scale > 0