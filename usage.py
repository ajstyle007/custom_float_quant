from custom_float_quant import CustomFloat

# --- Named presets ---
cf = CustomFloat.mxfp4_e2m1()
cf.quantize(3.14159265)

# or by string, useful for config-driven code:
cf = CustomFloat.from_preset("bf16")

# --- Custom formats: not limited to the 10 named presets ---
# CustomFloat is a general (exponent_bits, mantissa_bits) simulator - the
# presets above are just convenience shortcuts for well-known formats.
# Construct any split you want to explore directly:
cf = CustomFloat(exponent_bits=6, mantissa_bits=9)   # a made-up 16-bit format
cf.quantize(3.14159265)

# Bias is derived automatically using the standard IEEE formula:
# bias = 2**(exponent_bits - 1) - 1
print(cf.bias)             # 31, for exponent_bits=6
print(cf.max_finite_value) # largest finite magnitude this format can hold

# Rounding mode and seed work identically to presets:
cf_narrow = CustomFloat(exponent_bits=3, mantissa_bits=2, rounding="stochastic", seed=7)
cf_narrow.quantize(3.14159265)



from custom_float_quant import CustomFloat
import numpy as np

cf = CustomFloat.mxfp4_e2m1()
weights = np.random.randn(4096, 4096)  # your actual model tensor shape
quantized = cf.quantize_array(weights)  # same shape, elementwise quantized

# quantize_array works identically for custom (non-preset) formats too:
cf_custom = CustomFloat(exponent_bits=5, mantissa_bits=4)
quantized_custom = cf_custom.quantize_array(weights)



from custom_float_quant import CustomFloat
import numpy as np

# Deterministic, cheap
cf = CustomFloat.mxfp8_e4m3(rounding="truncate")

# Unbiased, for QAT-style workflows — reproducible with a seed
cf = CustomFloat.mxfp8_e4m3(rounding="stochastic", seed=42)
weights_q = cf.quantize_array(weights)

# Still the default
cf = CustomFloat.mxfp8_e4m3()  # rounding="rne"



# --- Quantizing a real model (requires the [llm] extra: torch + transformers) ---
from custom_float_quant import CustomFloat
from custom_float_quant.vectorized import quantize_model_copy, evaluate_loss, perplexity_from_loss
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
cf = CustomFloat.mxfp8_e4m3()

# Non-destructive: returns a quantized deep copy, original model untouched
quantized_model = quantize_model_copy(model, cf, verbose=True)