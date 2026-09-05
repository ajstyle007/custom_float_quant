from custom_float_quant import PRESETS
from custom_float_quant.presets import FormatSpec


def test_all_presets_have_valid_specs():
    for name, spec in PRESETS.items():
        assert isinstance(spec, FormatSpec)
        assert spec.name == name
        assert spec.exponent_bits >= 1
        assert spec.mantissa_bits >= 1
        assert isinstance(spec.description, str) and len(spec.description) > 0


def test_expected_presets_present():
    expected = {
        "bf16", "fp16", "tf32",
        "mxfp8_e4m3", "mxfp8_e5m2",
        "mxfp6_e3m2", "mxfp6_e2m3",
        "mxfp4_e2m1",
        "fp8_e3m4", "fp8_e4m3fn",
    }
    assert set(PRESETS.keys()) == expected