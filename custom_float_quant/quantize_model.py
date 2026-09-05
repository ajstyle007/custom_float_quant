"""
Quantize a torch model's weights in-place through a CustomFloat format and
measure the resulting loss/perplexity impact vs the unquantized baseline.
"""

import copy

import torch
import numpy as np


def quantize_model_(model, cf, skip_layernorm=True, skip_bias=True, verbose=False,
                    use_block_scaling=False, block_size=32, power_of_two_scale=True):
    total_params = 0
    quantized_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += param.numel()

            if skip_layernorm and ("ln" in name or "layernorm" in name.lower() or "norm" in name.lower()):
                if verbose:
                    print(f"  skip (norm)  : {name}  {tuple(param.shape)}")
                continue
            if skip_bias and name.endswith(".bias"):
                if verbose:
                    print(f"  skip (bias)  : {name}  {tuple(param.shape)}")
                continue

            original = param.detach().cpu().numpy().astype(np.float64)
            
            if use_block_scaling:
                quantized, scale = cf.quantize_array_block_scaled(
                    original, block_size=block_size, power_of_two_scale=power_of_two_scale
                )
            else:
                quantized, scale = cf.quantize_array_scaled(original)
                quantized = quantized.astype(np.float32)
                param.copy_(torch.from_numpy(quantized))

            quantized_params += param.numel()
            if verbose:
                err = np.abs(original - quantized).mean()
                print(f"  quantize     : {name}  {tuple(param.shape)}  mean_abs_err={err:.6g}  scale={scale:.4g}")

    if verbose:
        pct = 100 * quantized_params / total_params
        print(f"Quantized {quantized_params:,}/{total_params:,} params ({pct:.1f}%)")

    return model

def quantize_model_copy(model, cf, **kwargs):
    """Non-destructive version: returns a quantized deep copy, leaves `model` untouched."""
    return quantize_model_(copy.deepcopy(model), cf, **kwargs)

@torch.no_grad()
def evaluate_loss(model, input_ids, targets):
    """Simple next-token cross-entropy loss over a batch, for before/after comparison."""
    model.eval()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    return loss.item()

def perplexity_from_loss(loss):
    return float(np.exp(loss))