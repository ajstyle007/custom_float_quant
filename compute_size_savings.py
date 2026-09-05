"""
Compute actual (for fp8_e4m3fn) and theoretical (for every other preset)
on-disk size savings from quantization.
"""
import os

import torch
from transformers import GPT2LMHeadModel

from custom_float_quant import CustomFloat, PRESETS
from custom_float_quant.quantize_model import quantize_model_copy

torch.manual_seed(0)

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ---- Baseline: real fp32 size ----
total_params = sum(p.numel() for p in model.parameters())
skipped_params = sum(
    p.numel() for name, p in model.named_parameters()
    if "ln" in name or "bias" in name
)
quantizable_params = total_params - skipped_params

fp32_bytes_per_param = 4
baseline_size_bytes = total_params * fp32_bytes_per_param

print(f"\nTotal params      : {total_params:,}")
print(f"Quantizable params: {quantizable_params:,} (rest are LayerNorm/bias, kept fp32)")
print(f"Baseline fp32 size: {baseline_size_bytes / 1e6:.2f} MB\n")


# ---- REAL, verified: fp8_e4m3fn cast to native torch.float8_e4m3fn ----
print("=== Real on-disk test: fp8_e4m3fn -> torch.float8_e4m3fn ===")
cf = CustomFloat.from_preset("fp8_e4m3fn")
q_model = quantize_model_copy(model, cf, verbose=False)

# Cast every quantized (non-skipped) weight into PyTorch's native 1-byte
# float8 dtype. Values are already rounded to what e4m3 can represent, so
# this cast should be lossless relative to the simulation - it's just
# changing the storage container, not re-quantizing.
with torch.no_grad():
    for name, param in q_model.named_parameters():
        if "ln" in name or name.endswith(".bias"):
            continue
        param.data = param.data.to(torch.float8_e4m3fn)

real_dir = os.path.join("saved_models", "gpt2_real_float8_e4m3fn")
os.makedirs(real_dir, exist_ok=True)

# save_pretrained doesn't support float8 tensors directly in older
# transformers versions - save the raw state_dict instead, which does.
torch.save(q_model.state_dict(), os.path.join(real_dir, "pytorch_model_fp8.bin"))

real_size_bytes = os.path.getsize(os.path.join(real_dir, "pytorch_model_fp8.bin"))
print(f"  Saved to: {real_dir}")
print(f"  Actual file size : {real_size_bytes / 1e6:.2f} MB")
print(f"  Baseline fp32 size: {baseline_size_bytes / 1e6:.2f} MB")
print(f"  Actual reduction  : {(1 - real_size_bytes / baseline_size_bytes) * 100:.1f}%")


# ---- THEORETICAL: every preset, based on bits-per-value ----
print("\n=== Theoretical size (bits-per-value math, no native dtype needed) ===")
print(f"{'format':16} {'bits':>6} {'theoretical size':>18} {'reduction':>12}")

for name in PRESETS:
    spec_bits = 1 + PRESETS[name].exponent_bits + PRESETS[name].mantissa_bits  # sign + exp + mantissa
    quantized_bytes = quantizable_params * spec_bits / 8  # fractional bytes are fine here, it's a size estimate
    kept_fp32_bytes = skipped_params * fp32_bytes_per_param
    theoretical_total = quantized_bytes + kept_fp32_bytes
    reduction_pct = (1 - theoretical_total / baseline_size_bytes) * 100
    print(f"{name:16} {spec_bits:>6} {theoretical_total / 1e6:>15.2f} MB {reduction_pct:>11.1f}%")

print(
    "\nNote: only fp8_e4m3fn above was actually verified on disk (native "
    "PyTorch dtype). The rest are size estimates from bit-width math - "
    "getting real files that small for mxfp6/mxfp4 would require manual "
    "bit-packing, since PyTorch has no native 6-bit or 4-bit tensor type."
)


# Total params      : 124,439,808
# Quantizable params: 124,318,464 (rest are LayerNorm/bias, kept fp32)
# Baseline fp32 size: 497.76 MB

# === Real on-disk test: fp8_e4m3fn -> torch.float8_e4m3fn ===
#   Saved to: saved_models\gpt2_real_float8_e4m3fn
#   Actual file size : 124.86 MB
#   Baseline fp32 size: 497.76 MB
#   Actual reduction  : 74.9%


# === Theoretical size (bits-per-value math, no native dtype needed) ===
# format             bits   theoretical size    reduction

# bf16                 16          249.12 MB        50.0%
# fp16                 16          249.12 MB        50.0%
# tf32                 19          295.74 MB        40.6%
# mxfp8_e4m3            8          124.80 MB        74.9%
# mxfp8_e5m2            8          124.80 MB        74.9%
# mxfp6_e3m2            6           93.72 MB        81.2%
# mxfp6_e2m3            6           93.72 MB        81.2%
# mxfp4_e2m1            4           62.64 MB        87.4%
# fp8_e3m4              8          124.80 MB        74.9%
# fp8_e4m3fn            8          124.80 MB        74.9%

# Note: only fp8_e4m3fn above was actually verified on disk (native PyTorch dtype). 
# The rest are size estimates from bit-width math - getting real files that small 
# for mxfp6/mxfp4 would require manual bit-packing, since PyTorch has no native 6-bit or 4-bit tensor type.