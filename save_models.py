"""
Save the original GPT-2 model and the best-performing quantized variant
to disk, so they can be loaded independently later for manual testing.
"""

import json
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from custom_float_quant import CustomFloat
from custom_float_quant.quantize_model import quantize_model_copy

torch.manual_seed(0)

BEST_PRESET = "fp8_e3m4"

OUTPUT_DIR = os.path.join("saved_models")
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "gpt2_original")
QUANTIZED_DIR = os.path.join(OUTPUT_DIR, f"gpt2_quantized_{BEST_PRESET}")

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

print(f"Saving original model to: {ORIGINAL_DIR}")
os.makedirs(ORIGINAL_DIR, exist_ok=True)
model.save_pretrained(ORIGINAL_DIR)
tokenizer.save_pretrained(ORIGINAL_DIR)

print(f"Quantizing through preset: {BEST_PRESET}")
cf = CustomFloat.from_preset(BEST_PRESET)
q_model = quantize_model_copy(model, cf, verbose=True)


print(f"Saving quantized model to: {QUANTIZED_DIR}")
os.makedirs(QUANTIZED_DIR, exist_ok=True)
q_model.save_pretrained(QUANTIZED_DIR)
tokenizer.save_pretrained(QUANTIZED_DIR)

metadata = {
    "preset": BEST_PRESET,
    "exponent_bits": cf.exponent_bits,
    "mantissa_bits": cf.mantissa_bits,
    "rounding": cf.rounding,
    "scaling": "per-tensor",
    "note": (
        "Weights are stored as float32 - this simulates the format's "
        "precision loss (values are rounded to what the format could "
        "represent) but does NOT reduce file size on disk. Actual "
        "storage/inference savings would require real low-bit tensors "
        "and kernels, not just value rounding."
    ),
}

with open(os.path.join(QUANTIZED_DIR, "quantization_info.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nDone.")
print(f"  Original : {ORIGINAL_DIR}")
print(f"  Quantized: {QUANTIZED_DIR}")