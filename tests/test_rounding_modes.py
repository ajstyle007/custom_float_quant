import math
import random

import numpy as np

from custom_float_quant import CustomFloat, PRESETS


def test_truncate_never_rounds_up():
    cf_trunc = CustomFloat.from_preset("mxfp8_e4m3", rounding="truncate")
    cf_rne = CustomFloat.from_preset("mxfp8_e4m3", rounding="rne")

    shift = 52 - cf_trunc.mantissa_bits
    frac_bits = (1 << (shift - 1)) | (1 << (shift - 2))  # guard=1, round=1 -> RNE rounds up
    val = 1.0 + frac_bits / (1 << 52)

    t = cf_trunc.quantize(val)
    r = cf_rne.quantize(val)
    assert abs(t) <= abs(r)


def test_truncate_scalar_matches_vectorized(preset_name):
    random.seed(1)
    cf = CustomFloat.from_preset(preset_name, rounding="truncate")
    values = [random.uniform(-1e3, 1e3) for _ in range(200)]
    values += [0.0, -0.0, float("inf"), float("-inf"), float("nan")]

    scalar = [cf.quantize(v) for v in values]
    vector = cf.quantize_array(np.array(values, dtype=np.float64))

    for v, s, vec in zip(values, scalar, vector):
        if math.isnan(s) and math.isnan(vec):
            continue
        assert s == vec, f"value={v!r} scalar={s!r} vector={vec!r}"


def test_stochastic_reproducible_with_seed_scalar():
    cf_a = CustomFloat.from_preset("mxfp8_e4m3", rounding="stochastic", seed=123)
    cf_b = CustomFloat.from_preset("mxfp8_e4m3", rounding="stochastic", seed=123)

    seq_a = [cf_a.quantize(3.14159265) for _ in range(20)]
    seq_b = [cf_b.quantize(3.14159265) for _ in range(20)]
    assert seq_a == seq_b


def test_stochastic_reproducible_with_seed_vectorized():
    cf_a = CustomFloat.from_preset("mxfp8_e4m3", rounding="stochastic", seed=42)
    cf_b = CustomFloat.from_preset("mxfp8_e4m3", rounding="stochastic", seed=42)

    arr = np.full(5000, 3.14159265)
    out_a = cf_a.quantize_array(arr)
    out_b = cf_b.quantize_array(arr)
    assert np.array_equal(out_a, out_b)


def test_stochastic_is_unbiased_in_expectation():
    """Mean of many stochastic draws for a single value should land much
    closer to the true value than RNE's single deterministic estimate."""
    true_val = 3.14159265

    cf_stoch = CustomFloat.from_preset("mxfp8_e4m3", rounding="stochastic", seed=7)
    draws = np.array([cf_stoch.quantize(true_val) for _ in range(5000)])
    mean_stoch = draws.mean()

    cf_rne = CustomFloat.from_preset("mxfp8_e4m3", rounding="rne")
    rne_val = cf_rne.quantize(true_val)

    stoch_error = abs(mean_stoch - true_val)
    rne_error = abs(rne_val - true_val)

    assert stoch_error < rne_error