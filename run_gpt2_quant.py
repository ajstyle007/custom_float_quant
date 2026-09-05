"""
Quantize GPT-2 through each CustomFloat preset and compare perplexity
on a small text sample against the fp32 baseline.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from custom_float_quant import CustomFloat, PRESETS
from custom_float_quant.quantize_model import quantize_model_copy, evaluate_loss, perplexity_from_loss

torch.manual_seed(0)

print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

text = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models often require significant computational resources. "
    "Quantization is a technique used to reduce model size and inference cost."
)

encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings["input_ids"]


# Next-token prediction: input is all tokens except the last, target is
# all tokens except the first (standard causal LM shift)
inputs = input_ids[:, :-1]
targets = input_ids[:, 1:]

def eval_gpt2(m):
    m.eval()
    with torch.no_grad():
        logits = m(inputs).logits
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    return loss.item()

baseline_loss = eval_gpt2(model)
baseline_ppl = perplexity_from_loss(baseline_loss)

print(f"\n{'format':16} {'loss':>10} {'perplexity':>12} {'delta_ppl':>12}")
print(f"{'baseline (fp32)':16} {baseline_loss:>10.4f} {baseline_ppl:>12.4f} {'--':>12}")


for name in PRESETS:
    cf = CustomFloat.from_preset(name)
    q_model = quantize_model_copy(model, cf, verbose=False)
    loss = eval_gpt2(q_model)
    ppl = perplexity_from_loss(loss)
    delta = ppl - baseline_ppl
    print(f"{name:16} {loss:>10.4f} {ppl:>12.4f} {delta:>+12.4f}")


#Output
# format                 loss   perplexity    delta_ppl

# baseline (fp32)      4.5786      97.3742           --
# bf16                 4.5889      98.3839      +1.0097
# fp16                 4.5756      97.0886      -0.2856
# tf32                 4.5756      97.0886      -0.2856
# mxfp8_e4m3           4.5893      98.4286      +1.0545
# mxfp8_e5m2           4.6035      99.8335      +2.4594
# mxfp6_e3m2           4.7092     110.9663     +13.5922
# mxfp6_e2m3           8.5355    5092.1952   +4994.8210
# mxfp4_e2m1           9.6496   15515.9954  +15418.6212
# fp8_e3m4             4.5704      96.5817      -0.7925
# fp8_e4m3fn           4.5893      98.4286      +1.0545