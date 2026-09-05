import math
import struct
from dataclasses import dataclass


@dataclass
class FPNumber:
    sign: int
    exponent: int
    mantissa: int
    is_zero: bool = False
    is_subnormal: bool = False
    is_inf: bool = False
    is_nan: bool = False


# def float_to_uint32(value: float) -> int:
#     """Convert a Python float to its IEEE754 32-bit integer representation."""
#     return struct.unpack(">I", struct.pack(">f", value))[0]

def float_to_uint64(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def uint32_to_float(bits: int) -> float:
    """Convert IEEE754 32-bit integer back to float."""
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def bits_str(value: int, width: int) -> str:
    return format(value, f"0{width}b")


def decode_fp64(value: float):
    """Decode an FP32 number into sign, exponent and mantissa."""
    bits = float_to_uint64(value)

    sign = (bits >> 63) & 0x1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF  # 52 mantissa bits
    unbiased_exponent = exponent - 1023

    return {
        "value": value,
        "bits": bits_str(bits, 32),
        "sign": sign,
        "exponent_bits": bits_str(exponent, 8),
        "exponent": exponent,
        "unbiased_exponent": unbiased_exponent,
        "mantissa_bits": bits_str(mantissa, 23),
        "mantissa": mantissa,
        "is_zero": exponent == 0 and mantissa == 0,
        "is_subnormal": exponent == 0 and mantissa != 0,
        "is_inf": exponent == 255 and mantissa == 0,
        "is_nan": exponent == 255 and mantissa != 0,
    }


class CustomFloat:

    def __init__(self, exponent_bits, mantissa_bits, rounding="rne"):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.rounding = rounding

        self.bias = (1 << (exponent_bits - 1)) - 1
        self.max_exponent = (1 << exponent_bits) - 1
        self.max_normal_exponent = self.max_exponent - 1

    def encode_exponent(self, unbiased_exp):
        stored = unbiased_exp + self.bias

        if stored <= 0:
            return 0

        if stored > self.max_normal_exponent:
            stored = self.max_exponent

        return stored

    def extract_grs(self, mantissa):
        # shift = 23 - self.mantissa_bits
        shift = 52 - self.mantissa_bits

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
        grs = self.extract_grs(mantissa)

        kept = grs["kept"]
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

            # mantissa = (1 << 23) | info["mantissa"]  # restore hidden 1
            mantissa = (1 << 52) | info["mantissa"]
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


# if __name__ == "__main__":
#     cf = CustomFloat(7, 8)

#     tests = [
#         3.14159265,
#         -8.75,
#         0.123456,
#         100.125,
#         2**-62,
#         2**-63,
#         2**-70,
#         1e20,
#         1e30,
#         3.4e38,
#         -0.000123,
#         -12345.678,
#         0.0,
#         -0.0,
#         float("inf"),
#         float("-inf"),
#         float("nan"),
#     ]

#     for x in tests:
#         y = cf.quantize(x)
#         error = abs(x - y) if math.isfinite(x) and math.isfinite(y) else "N/A"
#         print(f"{x!r:>25} -> {y!r:<25} error={error}")