# custom_float_quant
A from-scratch IEEE-754-style floating point simulator for exploring low-precision number formats used in LLM quantization (FP8, FP6, FP4, bfloat16, TF32, and the OCP Microscaling family) - built to understand why these formats work the way they do, not just to call a library that already does it.

📖 Read the full writeup: What Actually Happens When You Quantize an LLM?

<img width="1408" height="768" alt="quantize_llm" src="https://github.com/user-attachments/assets/6b9bb64c-15c3-4361-a403-c7cd68529be2" />


### What this is

Every format in this library is implemented at the bit level: sign, exponent, mantissa, biasing, subnormals, infinities, and NaN - built up from IEEE-754 binary64 decoding rather than wrapping an existing quantization library. The goal was to actually understand quantization mechanics (guard/round/sticky rounding, round-to-nearest-even tie breaking, exponent overflow, subnormal underflow) well enough to explain every number it produces, then use that understanding to quantize a real language model (GPT-2) and measure the real accuracy/size tradeoff.

### Features
- 10 named format presets covering general-purpose training formats (bf16, fp16, tf32) and the OCP Microscaling family used on current accelerator hardware (MXFP8, MXFP6, MXFP4 / NVFP4), plus additional FP8 splits from quantization research
- Three rounding modes: round-to-nearest-even (default), truncate, and stochastic rounding (with seeded reproducibility)
- Vectorized batch quantization (quantize_array) using real numpy bit operations - not a Python loop - validated bit-for-bit identical to the scalar path across thousands of constructed test cases
- Per-tensor scaling (quantize_array_scaled) to keep narrow formats (FP4/FP6) from overflowing to inf on real weight distributions
- Per-block scaling (quantize_array_block_scaled) matching real OCP Microscaling hardware - each 32-element block gets its own scale, so one outlier only costs precision in its own neighborhood instead of the whole tensor. Optionally restrict the scale to a power of two (power_of_two_scale=True, matching real hardware's E8M0 format) or keep it continuous to measure what that hardware constraint costs
- A working end-to-end LLM quantization pipeline: load a real HuggingFace model, quantize its weights through any preset, and measure the perplexity impact against the fp32 baseline

### Installation
```
pip install custom-float-quant
```
For the end-to-end model-quantization pipeline (requires torch and transformers):
```
pip install "custom-float-quant[llm]"
```

### Format presets
Preset	Bits (sign+exp+mantissa)	Notes
bf16	1+8+7	Default mixed-precision training format
fp16	1+5+10	IEEE half precision
tf32	1+8+10	NVIDIA tensor-core training format
mxfp8_e4m3	1+4+3	OCP standard, precision-favoring FP8
mxfp8_e5m2	1+5+2	OCP standard, range-favoring FP8 (gradients)
mxfp6_e3m2	1+3+2	OCP MXFP6, range-favoring
mxfp6_e2m3	1+2+3	OCP MXFP6, precision-favoring
mxfp4_e2m1	1+2+1	OCP MXFP4 / NVIDIA NVFP4, native on Blackwell
fp8_e3m4	1+3+4	Mantissa-favoring FP8, weight-quantization research
fp8_e4m3fn	1+4+3	Finite-only OCP variant of e4m3

The presets above are convenience shortcuts, not the limit of what this library does. CustomFloat is a general (exponent_bits, mantissa_bits) simulator - construct any split you want to explore directly:

```
from custom_float_quant import CustomFloat

# A made-up 16-bit format: 1 sign + 6 exponent + 9 mantissa bits
cf = CustomFloat(exponent_bits=6, mantissa_bits=9)
cf.quantize(3.14159265)

# Bias is derived automatically: bias = 2**(exponent_bits - 1) - 1
cf.bias                # 31
cf.max_finite_value    # largest finite magnitude this format can hold

# Rounding mode and seed work exactly the same as for presets
cf = CustomFloat(exponent_bits=3, mantissa_bits=2, rounding="stochastic", seed=7)
```

This is the core use case for exploring formats that don't have an established name yet - e.g. testing whether a 5-bit exponent / 4-bit mantissa split holds up better than MXFP4 on a particular weight distribution, before anyone has standardized it.

```
from custom_float_quant import CustomFloat
import numpy as np

# Scalar
cf = CustomFloat.mxfp8_e4m3()
cf.quantize(3.14159265)  # -> 3.25

# Vectorized (real model tensors, not just scalars)
weights = np.random.randn(4096, 4096)
quantized = cf.quantize_array(weights)

# Rounding modes
cf = CustomFloat.mxfp8_e4m3(rounding="stochastic", seed=42)  # unbiased in expectation
cf = CustomFloat.mxfp8_e4m3(rounding="truncate")              # cheap, deterministic, biased

# Per-tensor scaling - required for narrow formats (FP4/FP6) to avoid
# overflowing to inf on real weight magnitudes
quantized, scale = cf.quantize_array_scaled(weights)

# Per-block scaling (new in 0.3.0) - matches how real OCP Microscaling
# hardware quantizes narrow formats: each block of 32 elements gets its
# own scale, so one outlier only costs precision in its own block instead
# of dragging down the whole tensor's precision.
quantized, scales = cf.quantize_array_block_scaled(weights, block_size=32)

# power_of_two_scale=True (default) restricts each block's scale to a
# power of two, matching real hardware's E8M0 shared-exponent scale.
# Set False to see how much accuracy that hardware constraint costs.
quantized_ideal, _ = cf.quantize_array_block_scaled(weights, power_of_two_scale=False)
```

#### Quantizing a real model
```
from transformers import GPT2LMHeadModel
from custom_float_quant import CustomFloat
from custom_float_quant.quantize_model import quantize_model_copy, evaluate_loss, perplexity_from_loss

model = GPT2LMHeadModel.from_pretrained("gpt2")
cf = CustomFloat.mxfp8_e4m3()
quantized_model = quantize_model_copy(model, cf, verbose=True)
```

#### Validated results (GPT-2, 124M params)
Perplexity on a fixed eval sample, fp32 baseline = 97.37:

Format	Perplexity	Δ vs baseline
bf16	98.38	+1.01
fp16	97.09	-0.29
tf32	97.09	-0.29
mxfp8_e4m3	98.43	+1.05
mxfp8_e5m2	99.83	+2.46
fp8_e3m4	96.58	-0.79
fp8_e4m3fn	98.43	+1.05
mxfp6_e3m2	110.97	+13.59
mxfp6_e2m3	5,092.20	+4,994.82
mxfp4_e2m1	15,515.99	+15,418.62

#### Project structure
```
custom_fp_quantization/
├── pyproject.toml
├── LICENSE
├── README.md
├── PUBLISHING.md
├── quick_demo.py
├── usage.py
├── custom_float_quant/
│   ├── __init__.py       # public API re-exports
│   ├── core.py            # CustomFloat class, scalar quantize/encode/decode
│   ├── presets.py          # named format presets
│   ├── vectorized.py       # numpy-vectorized batch quantization
│   └── quantize_model.py   # torch model quantization + eval helpers ([llm] extra)
└── tests/
    ├── conftest.py
    ├── test_core.py
    ├── test_presets.py
    ├── test_rounding_modes.py
    ├── test_vectorized.py
    └── test_block_scaling.py
```

#### License
MIT - see LICENSE.
