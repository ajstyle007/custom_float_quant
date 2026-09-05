"""
Named format presets for CustomFloat.
 
Bias for every preset follows the standard IEEE-style formula
bias = 2**(exponent_bits - 1) - 1, so only (exponent_bits, mantissa_bits)
need to be specified per format.
 
Selection covers the formats actually used across the LLM quantization
stack today: general-purpose training formats (bf16, fp16, tf32) and the
OCP Microscaling (MX) family that current accelerators (NVIDIA Blackwell,
AMD MI355X) natively support in hardware, plus two additional FP8 splits
that show up in weight-quantization research alongside the OCP-standard
E4M3/E5M2 pair.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class FormatSpec:
    name: str
    exponent_bits: int
    mantissa_bits: int
    description: str

PRESETS: dict[str, FormatSpec] = {

    # --- General-purpose training / inference formats ---
    "bf16": FormatSpec(
        "bf16", 8, 7,
        "Google Brain Float16. Same exponent range as FP32 (safe for "
        "training, no extra rescaling needed), coarse mantissa. The "
        "default mixed-precision training format on TPUs/GPUs."
    ),

    "fp16": FormatSpec(
        "fp16", 5, 10,
        "IEEE-754 half precision. Better mantissa precision than bf16 "
        "but a much narrower exponent range - prone to overflow on "
        "large-loss-scale training without careful scaling."
    ),

    "tf32": FormatSpec(
        "tf32", 8, 10,
        "NVIDIA TensorFloat-32. FP32 dynamic range with a truncated "
        "10-bit mantissa; used transparently by tensor cores for "
        "matmuls on Ampere+ GPUs, not usually stored, just computed in."
    ),

    # --- OCP Microscaling (MX) family: block-scaled, hardware-native on
    #     Blackwell-class GPUs. Element format below is paired with a
    #     shared per-block E8M0 exponent scale in real MX deployments;
    #     this simulator models the per-element format only. ---

    "mxfp8_e4m3": FormatSpec(
        "mxfp8_e4m3", 4, 3,
        "OCP MXFP8, E4M3 variant. Favors precision over range - the "
        "standard choice for FP8 weights/activations on the forward pass."
    ),
    "mxfp8_e5m2": FormatSpec(
        "mxfp8_e5m2", 5, 2,
        "OCP MXFP8, E5M2 variant. Favors dynamic range over precision - "
        "commonly used for gradients on the backward pass."
    ),
    "mxfp6_e3m2": FormatSpec(
        "mxfp6_e3m2", 3, 2,
        "OCP MXFP6, E3M2 variant. Range-favoring 6-bit format, a step "
        "down from MXFP8 with roughly proportional throughput gains."
    ),
    "mxfp6_e2m3": FormatSpec(
        "mxfp6_e2m3", 2, 3,
        "OCP MXFP6, E2M3 variant. Precision-favoring 6-bit format for "
        "blocks with a narrower dynamic range."
    ),
    "mxfp4_e2m1": FormatSpec(
        "mxfp4_e2m1", 2, 1,
        "OCP MXFP4 (E2M1). The 4-bit element format underlying MXFP4 "
        "and NVIDIA's NVFP4 - native on Blackwell tensor cores. Only "
        "8 nonzero magnitudes per sign; needs block scaling and "
        "outlier-aware calibration (e.g. Hadamard rotation, GPTQ-style "
        "reconstruction) to hold up on real LLM weights."
    ),

    # --- Additional FP8 splits from weight-quantization research,
    #     outside the two OCP-standard MXFP8 variants above ---

    "fp8_e3m4": FormatSpec(
        "fp8_e3m4", 3, 4,
        "8-bit float favoring mantissa precision over exponent range. "
        "Explored in FP8-format research as a better fit than E4M3 for "
        "weight-only quantization, where values are already narrow-range "
        "post-normalization."
    ),
    "fp8_e4m3fn": FormatSpec(
        "fp8_e4m3fn", 4, 3,
        "Same bit split as mxfp8_e4m3 (4 exponent, 3 mantissa bits). "
        "The 'fn' (finite-only) OCP variant reassigns the reserved "
        "all-ones exponent pattern to NaN instead of Inf, trading "
        "infinity representation for one extra representable magnitude. "
        "This simulator does not yet distinguish fn from standard "
        "saturate-to-inf behavior - see TODO.md."
    ),
}