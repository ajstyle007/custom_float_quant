# Let's first build utilities to inspect an FP32 number.

import struct

def float_to_uint32(value: float) -> int:
    """Convert a Python float to its IEEE754 32-bit integer representation."""
    return struct.unpack(">I", struct.pack(">f", value))[0]

# print(float_to_uint32(3.14159265))
# 1078530011

def uint32_to_float(bits: int) -> float:
    """Convert IEEE754 32-bit integer back to float."""
    return struct.unpack(">f", struct.pack(">I", bits))[0]

# print(uint32_to_float(1078530011))
# 3.1415927410125732


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
    }



info = decode_fp32(3.14159265)

for k, v in info.items():
    print(f"{k}: {v}")



def mantissa_as_float(mantissa: int) -> float:
    """
    Convert the 23 stored mantissa bits into a real value.

    Returns a number in [1,2)
    """

    return 1.0 + mantissa / (1 << 23)

print(mantissa_as_float(info["mantissa"]))



def reconstruct(sign, exponent, mantissa):

    fraction = mantissa_as_float(mantissa)

    value = ((-1) ** sign) * fraction * (2 ** exponent)

    return value


x = 3.14159265

info = decode_fp32(x)

y = reconstruct(
    info["sign"],
    info["unbiased_exponent"],
    info["mantissa"],
)

print(x)
print(y)


# output
# value: 3.14159265
# bits: 01000000010010010000111111011011
# sign: 0
# exponent_bits: 10000000
# exponent: 128
# unbiased_exponent: 1
# mantissa_bits: 10010010000111111011011
# mantissa: 4788187
# 1.5707963705062866
# 3.14159265
# 3.1415927410125732