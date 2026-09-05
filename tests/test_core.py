import math

import pytest

from custom_float_quant import CustomFloat, PRESETS


def test_all_presets_construct(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    assert cf.preset_name == preset_name


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        CustomFloat.from_preset("not_a_real_format")


def test_invalid_rounding_mode_raises():
    with pytest.raises(ValueError):
        CustomFloat(4, 3, rounding="banana")


def test_zero_roundtrip(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    assert cf.quantize(0.0) == 0.0
    assert math.copysign(1.0, cf.quantize(0.0)) == 1.0

    neg_zero = cf.quantize(-0.0)
    assert neg_zero == 0.0
    assert math.copysign(1.0, neg_zero) == -1.0


def test_inf_roundtrip(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    assert cf.quantize(float("inf")) == float("inf")
    assert cf.quantize(float("-inf")) == float("-inf")


def test_nan_roundtrip(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    assert math.isnan(cf.quantize(float("nan")))


def test_overflow_saturates_to_inf(preset_name):
    """A value far beyond any format's range should saturate to inf, not
    silently collide with a finite representable value."""
    cf = CustomFloat.from_preset(preset_name)
    assert cf.quantize(1e300) == float("inf")
    assert cf.quantize(-1e300) == float("-inf")


def test_subnormal_does_not_crash(preset_name):
    cf = CustomFloat.from_preset(preset_name)
    # Deliberately tiny value, well below any preset's normal range
    result = cf.quantize(2.0 ** -140)
    assert result >= 0.0
    assert math.isfinite(result) or result == 0.0


def test_named_classmethods_match_from_preset():
    assert CustomFloat.bf16().preset_name == "bf16"
    assert CustomFloat.fp16().preset_name == "fp16"
    assert CustomFloat.mxfp4_e2m1().preset_name == "mxfp4_e2m1"


def test_all_ten_presets_registered():
    assert len(PRESETS) == 10