import pytest

from custom_float_quant import PRESETS


@pytest.fixture(params=list(PRESETS.keys()))
def preset_name(request):
    """Parametrized fixture: every test using this runs once per preset."""
    return request.param