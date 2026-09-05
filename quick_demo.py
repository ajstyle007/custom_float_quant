from custom_float_quant import CustomFloat
import numpy as np

# Pick a coarse format so rounding differences are actually visible
value = 3.14159265

print("Rounding modes on a single value:")
for mode in ["rne", "truncate", "stochastic"]:
    cf = CustomFloat.mxfp8_e4m3(rounding=mode, seed=1)
    result = cf.quantize(value)
    print(f"  {mode:12} -> {result}")

print()
print("Stochastic is random - run it a few times:")
cf = CustomFloat.mxfp8_e4m3(rounding="stochastic", seed=1)
for _ in range(5):
    print(f"  {cf.quantize(value)}")

print()
print("Same seed -> reproducible:")
cf_a = CustomFloat.mxfp8_e4m3(rounding="stochastic", seed=99)
cf_b = CustomFloat.mxfp8_e4m3(rounding="stochastic", seed=99)
print(f"  {[cf_a.quantize(value) for _ in range(5)]}")
print(f"  {[cf_b.quantize(value) for _ in range(5)]}")

print()
print("Vectorized, on a small array:")
cf = CustomFloat.bf16()
arr = np.array([3.14159265, -8.75, 0.0001, 1e10])
print(f"  input : {arr}")
print(f"  output: {cf.quantize_array(arr)}")


print("="*100)

print("Custom (non-preset) formats - any exponent/mantissa split you want:")
cf_custom = CustomFloat(exponent_bits=6, mantissa_bits=9)
print(f"  CustomFloat(exponent_bits=6, mantissa_bits=9) -> {cf_custom.quantize(value)}")
print(f"  bias={cf_custom.bias}, max_finite_value={cf_custom.max_finite_value}")

import numpy as np

arr = np.random.randn(10000)

scalar = np.array([cf.quantize(x) for x in arr])
vector = cf.quantize_array(arr)

print(np.allclose(scalar, vector))
print(np.max(np.abs(scalar - vector)))