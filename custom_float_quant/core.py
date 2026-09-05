import random as _random
import struct
from dataclasses import dataclass

from .presets import PRESETS

ROUNDING_MODES = ("rne", "truncate", "stochastic")


@dataclass
class FPNumber:
    sign: int
    exponent: int
    mantissa: int
    is_zero: bool = False
    is_subnormal: bool = False
    is_inf: bool = False
    is_nan: bool = False


def float_to_uint64(value: float) -> int:
    """Convert a Python float to its IEEE754 64-bit integer representation."""
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def uint64_to_float(bits: int) -> float:
    """Convert IEEE754 64-bit integer back to a Python float."""
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def bits_str(value: int, width: int) -> str:
    return format(value, f"0{width}b")


# FP64 field widths, per IEEE-754 binary64
FP64_EXPONENT_BITS = 11
FP64_MANTISSA_BITS = 52
FP64_EXPONENT_BIAS = 1023
FP64_MAX_EXPONENT = (1 << FP64_EXPONENT_BITS) - 1  # 2047


def decode_fp64(value: float):
    """Decode a Python float (IEEE-754 binary64) into sign/exponent/mantissa."""
    bits = float_to_uint64(value)

    sign = (bits >> 63) & 0x1
    exponent = (bits >> FP64_MANTISSA_BITS) & FP64_MAX_EXPONENT
    mantissa = bits & ((1 << FP64_MANTISSA_BITS) - 1)
    unbiased_exponent = exponent - FP64_EXPONENT_BIAS

    return {
        "value": value,
        "bits": bits_str(bits, 64),
        "sign": sign,
        "exponent_bits": bits_str(exponent, FP64_EXPONENT_BITS),
        "exponent": exponent,
        "unbiased_exponent": unbiased_exponent,
        "mantissa_bits": bits_str(mantissa, FP64_MANTISSA_BITS),
        "mantissa": mantissa,
        "is_zero": exponent == 0 and mantissa == 0,
        "is_subnormal": exponent == 0 and mantissa != 0,
        "is_inf": exponent == FP64_MAX_EXPONENT and mantissa == 0,
        "is_nan": exponent == FP64_MAX_EXPONENT and mantissa != 0,
    }


class CustomFloat:

    def __init__(self, exponent_bits, mantissa_bits, rounding="rne", seed=None, _preset_name=None):
        if rounding not in ROUNDING_MODES:
            raise ValueError(
                f"Unknown rounding mode {rounding!r}. "
                f"Must be one of: {', '.join(ROUNDING_MODES)}"
            )

        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.rounding = rounding
        self.seed = seed
        self.preset_name = _preset_name

        self.bias = (1 << (exponent_bits - 1)) - 1
        self.max_exponent = (1 << exponent_bits) - 1
        self.max_normal_exponent = self.max_exponent - 1

        # Lazily created RNGs, only touched when rounding == "stochastic".
        # Kept separate per-backend (stdlib random for the scalar path,
        # numpy Generator for quantize_array) since they don't share a
        # sequence anyway - each is independently seeded/reproducible.
        self._py_random = _random.Random(seed)
        self._np_rng = None  # created on first vectorized stochastic call

    def __repr__(self):
        label = f" ({self.preset_name})" if self.preset_name else ""
        return (f"CustomFloat(exponent_bits={self.exponent_bits}, "
                f"mantissa_bits={self.mantissa_bits}){label}")

    @property
    def max_finite_value(self):
        """Largest finite magnitude this format can represent."""
        max_mantissa_fraction = 1.0 + (2**self.mantissa_bits - 1) / 2**self.mantissa_bits
        return max_mantissa_fraction * 2 ** (self.max_normal_exponent - self.bias)

    # ---- Preset construction -------------------------------------------

    @classmethod
    def from_preset(cls, name, rounding="rne", seed=None):
        """Build a CustomFloat from a named preset in presets.PRESETS."""
        try:
            spec = PRESETS[name]
        except KeyError:
            available = ", ".join(sorted(PRESETS))
            raise ValueError(
                f"Unknown preset {name!r}. Available presets: {available}"
            ) from None
        return cls(spec.exponent_bits, spec.mantissa_bits,
                    rounding=rounding, seed=seed, _preset_name=spec.name)

    @classmethod
    def bf16(cls, rounding="rne", seed=None):
        return cls.from_preset("bf16", rounding=rounding, seed=seed)

    @classmethod
    def fp16(cls, rounding="rne", seed=None):
        return cls.from_preset("fp16", rounding=rounding, seed=seed)

    @classmethod
    def tf32(cls, rounding="rne", seed=None):
        return cls.from_preset("tf32", rounding=rounding, seed=seed)

    @classmethod
    def mxfp8_e4m3(cls, rounding="rne", seed=None):
        return cls.from_preset("mxfp8_e4m3", rounding=rounding, seed=seed)

    @classmethod
    def mxfp8_e5m2(cls, rounding="rne", seed=None):
        return cls.from_preset("mxfp8_e5m2", rounding=rounding, seed=seed)

    @classmethod
    def mxfp6_e3m2(cls, rounding="rne", seed=None):
        return cls.from_preset("mxfp6_e3m2", rounding=rounding, seed=seed)

    @classmethod
    def mxfp6_e2m3(cls, rounding="rne", seed=None):
        return cls.from_preset("mxfp6_e2m3", rounding=rounding, seed=seed)

    @classmethod
    def mxfp4_e2m1(cls, rounding="rne", seed=None):
        return cls.from_preset("mxfp4_e2m1", rounding=rounding, seed=seed)

    @classmethod
    def fp8_e3m4(cls, rounding="rne", seed=None):
        return cls.from_preset("fp8_e3m4", rounding=rounding, seed=seed)

    @classmethod
    def fp8_e4m3fn(cls, rounding="rne", seed=None):
        return cls.from_preset("fp8_e4m3fn", rounding=rounding, seed=seed)
    

    # ---- Encoding internals ---------------------------------------------

    def encode_exponent(self, unbiased_exp):
        stored = unbiased_exp + self.bias

        if stored <= 0:
            return 0

        if stored > self.max_normal_exponent:
            stored = self.max_exponent

        return stored

    def extract_grs(self, mantissa):
        shift = FP64_MANTISSA_BITS - self.mantissa_bits

        kept = mantissa >> shift
        guard = (mantissa >> (shift - 1)) & 1
        round_bit = (mantissa >> (shift - 2)) & 1
        sticky_mask = (1 << (shift - 2)) - 1
        sticky = 1 if (mantissa & sticky_mask) != 0 else 0

        return {
            "kept": kept,
            "guard": guard,
            "round": round_bit,
            "sticky": sticky,
        }

    def quantize_mantissa(self, mantissa):
        shift = FP64_MANTISSA_BITS - self.mantissa_bits
        kept = mantissa >> shift

        if self.rounding == "truncate":
            return kept

        if self.rounding == "stochastic":
            dropped = mantissa & ((1 << shift) - 1)
            prob_round_up = dropped / (1 << shift)
            if self._py_random.random() < prob_round_up:
                return kept + 1
            return kept

        # Default: round-to-nearest-even (RNE)
        grs = self.extract_grs(mantissa)

        guard = grs["guard"]
        round_bit = grs["round"]
        sticky = grs["sticky"]

        # Less than half -> keep
        if guard == 0:
            return kept

        # Greater than half -> round up
        if round_bit == 1:
            return kept + 1

        # Greater than half (sticky) -> round up
        if sticky == 1:
            return kept + 1

        # Exactly half (tie) -> round to nearest even
        if kept & 1:
            return kept + 1

        return kept

    def encode_custom(self, value):
        info = decode_fp64(value)

        if info["is_zero"]:
            return FPNumber(
                sign=info["sign"],
                exponent=0,
                mantissa=0,
                is_zero=True,
            )

        if info["is_inf"]:
            return FPNumber(
                sign=info["sign"],
                exponent=self.max_exponent,
                mantissa=0,
                is_inf=True,
            )

        if info["is_nan"]:
            return FPNumber(
                sign=info["sign"],
                exponent=self.max_exponent,
                mantissa=1,
                is_nan=True,
            )

        sign = info["sign"]
        exponent = self.encode_exponent(info["unbiased_exponent"])

        if exponent == self.max_exponent:
            return FPNumber(
                sign=sign,
                exponent=self.max_exponent,
                mantissa=0,
                is_inf=True,
            )

        if exponent == 0:
            shift = 1 - (info["unbiased_exponent"] + self.bias)

            mantissa = (1 << FP64_MANTISSA_BITS) | info["mantissa"]  # restore hidden 1
            mantissa >>= shift
            mantissa = self.quantize_mantissa(mantissa)

            if mantissa == 0:
                return FPNumber(
                    sign=sign,
                    exponent=0,
                    mantissa=0,
                    is_zero=True,
                )

            return FPNumber(
                sign=sign,
                exponent=0,
                mantissa=mantissa,
                is_subnormal=True,
            )

        mantissa = self.quantize_mantissa(info["mantissa"])

        # Did rounding create one extra bit?
        if mantissa >= (1 << self.mantissa_bits):
            mantissa = 0
            exponent += 1

            if exponent > self.max_normal_exponent:
                return FPNumber(
                    sign=sign,
                    exponent=self.max_exponent,
                    mantissa=0,
                    is_inf=True,
                )

        return FPNumber(
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
        )

    def decode_custom(self, fp: FPNumber):
        if fp.is_zero:
            return -0.0 if fp.sign else 0.0

        if fp.is_nan:
            return float("nan")

        if fp.is_inf:
            return float("-inf") if fp.sign else float("inf")

        if fp.is_subnormal:
            unbiased = 1 - self.bias
            fraction = fp.mantissa / (1 << self.mantissa_bits)
        else:
            unbiased = fp.exponent - self.bias
            fraction = 1.0 + fp.mantissa / (1 << self.mantissa_bits)

        return ((-1) ** fp.sign) * fraction * (2 ** unbiased)

    def quantize(self, value):
        fp = self.encode_custom(value)
        return self.decode_custom(fp)

    def quantize_array(self, values):
        """
        Quantize a numpy array (or array-like) elementwise through this
        format, using vectorized bit operations instead of a Python loop.
        Requires numpy. Result matches calling .quantize() on each element.
        """
        from .vectorized import quantize_array as _quantize_array
        return _quantize_array(self, values)


    def quantize_array_scaled(self, values, margin=0.95):
        """
        Like quantize_array, but rescales the tensor first so its max value
        fits within this format's representable range, then rescales back
        after quantizing. Prevents overflow-to-inf on narrow formats when the
        tensor's natural magnitude exceeds what the format can hold.

        margin: safety factor (<1.0) to avoid landing exactly on the overflow
            boundary after floating-point rounding during the rescale.

        Returns (quantized_values, scale_used).
        """
        import numpy as np
        arr = np.asarray(values, dtype=np.float64)
        max_abs = np.abs(arr).max()

        if max_abs == 0.0:
            return arr.copy(), 1.0  # nothing to scale

        scale = (self.max_finite_value * margin) / max_abs
        scaled = arr * scale
        quantized_scaled = self.quantize_array(scaled)
        return quantized_scaled / scale, scale


    def quantize_array_block_scaled(self, values, block_size=32, axis=-1, margin=0.95, power_of_two_scale=True):
        """
        Like quantize_array_scaled, but computes an independent scale factor
        per block of block_size elements (default 32, matching the OCP
        Microscaling spec) instead of one scale for the whole tensor. This is
        what real MXFP4/MXFP6/MXFP8 hardware does - a local outlier only
        affects its own block's scale, not the entire tensor's.

        power_of_two_scale=True (default) restricts each block's scale to a
        power of two, matching real hardware's 8-bit-exponent-only scale
        format. Set False for a continuous scale (more accurate, not
        hardware-representative) to measure what the constraint costs.
        """
        from .vectorized import quantize_array_block_scaled as _quantize_array_block_scaled
        return _quantize_array_block_scaled(
            self, values, block_size=block_size, axis=axis,
            margin=margin, power_of_two_scale=power_of_two_scale,
        )

