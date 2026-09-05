import struct
from dataclasses import dataclass

@dataclass
class FPNumber:
    sign: int
    exponent: int
    mantissa: int
    is_zero: bool=False
    is_subnormal: bool=False
    is_inf: bool=False
    is_nan: bool=False

def float_to_uint32(value: float) -> int:
        """Convert a Python float to its IEEE754 32-bit integer representation."""
        return struct.unpack(">I", struct.pack(">f", value))[0]

def uint32_to_float(bits: int) -> float:
    """Convert IEEE754 32-bit integer back to float."""
    return struct.unpack(">f", struct.pack(">I", bits))[0]

def bits_str(value: int, width: int) -> str:
    return format(value, f"0{width}b")

def decode_fp32(value: float):
        """
        Decode an FP32 number into sign, exponent and mantissa.
        """

        bits = float_to_uint32(value)

        sign = (bits >> 31) & 0x1
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF

        unbiased_exponent = exponent - 127

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

    def __init__(self, exponent_bits, mantissa_bits, rounding="truncate"):

        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.rounding = rounding

        self.bias = (1 << (exponent_bits - 1)) - 1

        self.max_exponent = (1 << exponent_bits) - 1
        self.max_normal_exponent = self.max_exponent - 1


    def encode_exponent(self, unbiased_exp): 
        # TODO: Convert exponent overflow to Infinity

        stored = unbiased_exp + self.bias

        if stored <= 0:
            return 0

        if stored > self.max_normal_exponent:
            stored = self.max_exponent

        return stored
    


    def extract_grs(self, mantissa):
        shift = 23 - self.mantissa_bits

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

        # Case 1: Less than half
        if guard == 0:
            print("Decision : Less than half -> Keep")
            return kept

        # Case 2: Greater than half
        if round_bit == 1:
            print("Decision : Greater than half -> Round Up")
            return kept + 1

        # Case 3: Greater than half
        if sticky == 1:
            print("Decision : Sticky bit set -> Round Up")
            return kept + 1

        # Case 4: Exactly half (Tie)
        # Round to nearest EVEN
        if kept & 1:
            print("Decision : Tie -> Odd -> Round Up")
            return kept + 1

        print("\nMantissa Reduction")
        print("------------------")
        print(f"Original Mantissa : {mantissa:023b}")
        print(f"Kept             : {kept:0{self.mantissa_bits}b}")
        print(f"Guard            : {guard}")
        print(f"Round            : {round_bit}")
        print(f"Sticky           : {sticky}")

        return kept


    def encode_custom(self, value):

        info = decode_fp32(value)

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
        # mantissa = self.quantize_mantissa(info["mantissa"])
        if exponent == self.max_exponent:
            return FPNumber(
                sign=sign,
                exponent=self.max_exponent,
                mantissa=0,
                is_inf=True,
            )

        if exponent == 0:
            shift = 1 - (info["unbiased_exponent"] + self.bias)

            mantissa = (1 << 23) | info["mantissa"]   # Restore hidden 1
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

        print("\nAfter Quantization")
        print("------------------")
        print(f"Exponent : {exponent}")
        print(f"Mantissa : {mantissa}")

        # Did rounding create one extra bit?
        if mantissa >= (1 << self.mantissa_bits):

            print("\nMantissa Overflow!")
            print("------------------")
            print("Carry propagated to exponent")

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

        print("=" * 80)

        print(f"Original Value : {value}")

        info = decode_fp32(value)

        print("\nFP32 Representation")
        print("-------------------")
        print(f"Bits      : {info['bits']}")
        print(f"Sign      : {info['sign']}")
        print(f"Exponent  : {info['exponent']} ({info['exponent_bits']})")
        print(f"Unbiased  : {info['unbiased_exponent']}")
        print(f"Mantissa  : {info['mantissa_bits']}")

        fp = self.encode_custom(value)

        print("\nCustom FP Representation")
        print("------------------------")
        print(fp)

        decoded = self.decode_custom(fp)

        print("\nDecoded Value")
        print("-------------")
        print(decoded)

        print("\nAbsolute Error")
        print("--------------")
        print(abs(value - decoded))

        return decoded


# cf = CustomFloat(7,8)
# y = cf.quantize(3.14159265)
# print(y)

cf = CustomFloat(7,8)

# tests = [float("inf"), float("-inf"), 0.0, -0.0, 3.14159265, -8.75, 2**-62, 2**-63, 2**-70,
#         2**-62,
#         2**-63,
#         2**-64,
#         2**-65,
#         2**-66,
#         2**-67,
#         2**-68,
#         2**-69,
#         2**-70,]

tests = [

    # Normal
    3.14159265,
    -8.75,
    0.123456,
    100.125,

    # Very small
    2**-62,
    2**-63,
    2**-70,

    # Very large
    1e20,
    1e30,
    3.4e38,

    # Negative
    -0.000123,
    -12345.678,

    # Edge values
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]

import math

for x in tests:
    y = cf.quantize(x)

    print()
    print("Original :", x)
    print("Decoded  :", y)

    print("\nAbsolute Error")
    print("--------------")

    if math.isfinite(x) and math.isfinite(y):
        print(abs(x - y))
    else:
        print("N/A")