"""
Vectorized (numpy) batch quantization.

Mirrors the scalar encode_custom/decode_custom logic in core.py bit-for-bit,
including its specific subnormal rounding behavior (see note in
_denormalize_mantissa below), so that CustomFloat.quantize_array(arr) always
agrees element-wise with calling CustomFloat.quantize(x) on each element.

Requires numpy. Imported lazily so numpy stays an optional dependency for
callers who only use the scalar API.
"""

import numpy as np

FP64_MANTISSA_BITS = 52
FP64_EXPONENT_MASK = np.uint64(0x7FF)
FP64_MANTISSA_MASK = np.uint64((1 << 52) - 1)
FP64_HIDDEN_BIT = np.uint64(1 << 52)


def _grs_round(value_bits, shift):
    """
    Guard/round/sticky rounding with round-to-nearest-even, vectorized.

    value_bits: uint64 ndarray - the raw bits to round.
    shift: plain Python int (constant across the whole array) - number of
           low bits being dropped. Must be >= 2.

    Returns an int64 ndarray of the rounded, right-shifted value.
    """
    kept = (value_bits >> np.uint64(shift)).astype(np.int64)
    guard = ((value_bits >> np.uint64(shift - 1)) & np.uint64(1)).astype(np.int64)
    round_bit = ((value_bits >> np.uint64(shift - 2)) & np.uint64(1)).astype(np.int64)
    sticky_mask = np.uint64((1 << (shift - 2)) - 1)
    sticky = ((value_bits & sticky_mask) != 0).astype(np.int64)

    tie_round_up = (round_bit == 0) & (sticky == 0) & ((kept & 1) == 1)
    round_up = (guard == 1) & ((round_bit == 1) | (sticky == 1) | tie_round_up)

    return kept + round_up.astype(np.int64)


def _truncate_round(value_bits, shift):
    """Plain truncation - drop the low `shift` bits, no rounding at all."""
    return (value_bits >> np.uint64(shift)).astype(np.int64)


def _stochastic_round(cf, value_bits, shift):
    """
    Stochastic rounding, vectorized: round up with probability equal to
    the fraction of the quantization step represented by the dropped bits,
    so the expectation over many samples is unbiased (unlike RNE, which is
    deterministic per value).
    """
    if cf._np_rng is None:
        cf._np_rng = np.random.default_rng(cf.seed)

    kept = (value_bits >> np.uint64(shift)).astype(np.int64)
    dropped_mask = np.uint64((1 << shift) - 1)
    dropped = (value_bits & dropped_mask).astype(np.float64)
    prob_round_up = dropped / float(1 << shift)

    draw = cf._np_rng.random(value_bits.shape)
    round_up = draw < prob_round_up

    return kept + round_up.astype(np.int64)


def _round_mantissa(cf, value_bits, shift):
    """Dispatch to the configured rounding mode for this CustomFloat."""
    if cf.rounding == "truncate":
        return _truncate_round(value_bits, shift)
    if cf.rounding == "stochastic":
        return _stochastic_round(cf, value_bits, shift)
    return _grs_round(value_bits, shift)


def _denormalize_mantissa(full_mantissa, denorm_shift):
    """
    Pre-shift a subnormal's full (hidden-bit-restored) mantissa by
    denorm_shift bits of plain truncation, matching core.py's scalar
    behavior: `mantissa >>= shift` happens BEFORE GRS rounding is applied,
    not as part of it.

    Note: this means bits dropped in this truncation step are not seen by
    the GRS sticky calculation that follows - the same double-rounding
    approximation the scalar path has. Kept identical on purpose so scalar
    and vectorized results always match; a "single combined shift" version
    would be more numerically correct but would diverge from core.py.
    """
    denorm_shift = denorm_shift.astype(np.int64)
    safe_shift = np.clip(denorm_shift, 0, 63).astype(np.uint64)
    shifted = full_mantissa >> safe_shift
    # Anything shifted by more than 63 bits is unambiguously zero.
    return np.where(denorm_shift > 63, np.uint64(0), shifted).astype(np.uint64)


def quantize_array(cf, values):
    """
    Quantize a numpy array (or array-like) through the given CustomFloat
    format, elementwise. Returns a float64 ndarray of the same shape.

    Equivalent to np.vectorize(cf.quantize)(values) but implemented with
    real numpy bit operations instead of a Python-level loop, so it scales
    to full model tensors.
    """
    arr = np.asarray(values, dtype=np.float64)
    shape = arr.shape
    flat = np.ascontiguousarray(arr.reshape(-1))
    bits = flat.view(np.uint64)

    sign = (bits >> np.uint64(63)) & np.uint64(1)
    exponent = (bits >> np.uint64(FP64_MANTISSA_BITS)) & FP64_EXPONENT_MASK
    mantissa_in = bits & FP64_MANTISSA_MASK
    unbiased = exponent.astype(np.int64) - 1023

    is_zero = (exponent == 0) & (mantissa_in == 0)
    is_inf_in = (exponent == 2047) & (mantissa_in == 0)
    is_nan_in = (exponent == 2047) & (mantissa_in != 0)
    is_finite_nonzero = ~(is_zero | is_inf_in | is_nan_in)

    stored = np.where(is_finite_nonzero, unbiased + cf.bias, np.int64(0))

    overflow_mask = is_finite_nonzero & (stored > cf.max_normal_exponent)
    subnormal_mask = is_finite_nonzero & (stored <= 0)
    normal_mask = is_finite_nonzero & ~overflow_mask & ~subnormal_mask

    out_exponent = np.zeros(flat.shape, dtype=np.int64)
    out_mantissa = np.zeros(flat.shape, dtype=np.int64)

    # ---- Normal path ----
    base_shift = FP64_MANTISSA_BITS - cf.mantissa_bits
    normal_rounded = _round_mantissa(cf, mantissa_in, base_shift)

    mantissa_carry = normal_mask & (normal_rounded >= (1 << cf.mantissa_bits))
    normal_exponent_out = np.where(mantissa_carry, stored + 1, stored)
    normal_mantissa_out = np.where(mantissa_carry, 0, normal_rounded)

    carry_overflow_mask = mantissa_carry & (normal_exponent_out > cf.max_normal_exponent)
    final_normal_mask = normal_mask & ~carry_overflow_mask
    final_overflow_mask = overflow_mask | carry_overflow_mask

    out_exponent = np.where(final_normal_mask, normal_exponent_out, out_exponent)
    out_mantissa = np.where(final_normal_mask, normal_mantissa_out, out_mantissa)

    # ---- Overflow path -> saturate to inf ----
    out_is_inf = is_inf_in | final_overflow_mask

    # ---- Subnormal path ----
    denorm_shift = 1 - stored
    full_mantissa = FP64_HIDDEN_BIT | mantissa_in
    shifted = _denormalize_mantissa(full_mantissa, denorm_shift)
    sub_rounded = _round_mantissa(cf, shifted, base_shift)

    sub_is_zero = subnormal_mask & (sub_rounded == 0)
    sub_is_subnormal = subnormal_mask & (sub_rounded != 0)

    out_mantissa = np.where(sub_is_subnormal, sub_rounded, out_mantissa)

    out_is_zero = is_zero | sub_is_zero
    out_is_subnormal = sub_is_subnormal

    # ---- Decode back to float64 ----
    sign_mult = np.where(sign == 1, -1.0, 1.0)

    zero_vals = np.where(sign == 1, -0.0, 0.0)
    nan_vals = np.full(flat.shape, np.nan)
    inf_vals = np.where(sign == 1, -np.inf, np.inf)

    sub_fraction = out_mantissa.astype(np.float64) / (1 << cf.mantissa_bits)
    sub_unbiased = 1 - cf.bias
    sub_vals = sign_mult * sub_fraction * (2.0 ** sub_unbiased)

    norm_fraction = 1.0 + out_mantissa.astype(np.float64) / (1 << cf.mantissa_bits)
    norm_unbiased = (out_exponent - cf.bias).astype(np.float64)
    norm_vals = sign_mult * norm_fraction * np.exp2(norm_unbiased)

    result = np.select(
        [is_nan_in, out_is_inf, out_is_zero, out_is_subnormal],
        [nan_vals, inf_vals, zero_vals, sub_vals],
        default=norm_vals,
    )

    return result.reshape(shape)


def quantize_array_block_scaled(cf, values, block_size=32, axis=-1, margin=0.95, power_of_two_scale=True):
    """
    Block-scaled quantization: splits values along axis into chunks of
    block_size elements, computes an independent scale factor per block
    (rather than one scale for the whole array), quantizes each block
    through cf, then rescales back.

    Matches the OCP Microscaling (MX) spec's approach: block_size=32,
    power_of_two_scale=True gives each block an 8-bit-exponent-only scale
    (no mantissa) - exactly what real MXFP4/MXFP6/MXFP8 hardware uses.
    Setting power_of_two_scale=False uses a continuous (non-power-of-two)
    scale instead - not representative of real hardware, but useful for
    isolating how much accuracy the power-of-two constraint itself costs.

    Returns (quantized_values, scales). quantized_values has the same
    shape as the input. scales has the same shape as the input except
    axis is replaced by the number of blocks along that axis.
    """
    arr = np.asarray(values, dtype=np.float64)
    orig_shape = arr.shape

    # Move the target axis to the end so we can reshape it into blocks
    arr = np.moveaxis(arr, axis, -1)
    n = arr.shape[-1]

    pad = (-n) % block_size
    if pad:
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, pad)]
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0.0)

    n_padded = arr.shape[-1]
    n_blocks = n_padded // block_size

    blocks = arr.reshape(*arr.shape[:-1], n_blocks, block_size)

    max_abs = np.abs(blocks).max(axis=-1, keepdims=True)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)  # avoid div-by-zero on all-zero blocks

    scale = (cf.max_finite_value * margin) / max_abs

    if power_of_two_scale:
        # Round DOWN to the nearest power of two so the scaled block never
        # exceeds the format's range (rounding up could push values past
        # max_finite_value after scaling). Matches real MX hardware's
        # 8-bit-exponent-only (E8M0) scale representation.
        scale = np.exp2(np.floor(np.log2(scale)))

    scaled_blocks = blocks * scale
    quantized_blocks = quantize_array(cf, scaled_blocks) / scale

    result = quantized_blocks.reshape(*arr.shape[:-1], n_padded)
    if pad:
        result = result[..., :n]
    result = np.moveaxis(result, -1, axis)

    scales = np.moveaxis(scale.squeeze(-1), -1, axis)

    return result.reshape(orig_shape), scales