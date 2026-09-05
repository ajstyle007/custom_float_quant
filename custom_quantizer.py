import struct

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
    }

info = decode_fp32(3.14159265)


def encode_exponent(unbiased_exponent: int, exponent_bits: int):
    """
    Encode an unbiased exponent using the given number of exponent bits.
    """

    bias = (1 << (exponent_bits - 1)) - 1

    stored_exponent = unbiased_exponent + bias

    max_exp = (1 << exponent_bits) - 1

    if stored_exponent < 0:
        stored_exponent = 0

    if stored_exponent > max_exp:
        stored_exponent = max_exp

    return stored_exponent


# print(encode_exponent(1, 7))
# 64


def truncate_mantissa(mantissa: int, target_bits: int):
    """
    Truncate the FP32 mantissa to target_bits.
    """

    shift = 23 - target_bits

    return mantissa >> shift


m = info["mantissa"]

new_m = truncate_mantissa(m, 8)

# print(bits_str(new_m, 8))


def encode_custom_fp(sign, unbiased_exponent, mantissa, exponent_bits, mantissa_bits):

    exponent = encode_exponent(unbiased_exponent, exponent_bits)

    mantissa = truncate_mantissa(mantissa, mantissa_bits)

    return {
        "sign": sign,
        "exponent": exponent,
        "mantissa": mantissa,
    }


custom = encode_custom_fp(
    info["sign"],
    info["unbiased_exponent"],
    info["mantissa"],
    exponent_bits=7,
    mantissa_bits=8,
)


print(custom)

print(bits_str(custom["exponent"],7))
print(bits_str(custom["mantissa"],8))



def decode_custom_fp(sign: int, exponent: int, mantissa: int, exponent_bits: int, mantissa_bits: int):
    """
    Decode a custom floating-point number back to FP32.
    """

    # Calculate bias for this custom format
    bias = (1 << (exponent_bits - 1)) - 1

    # Recover unbiased exponent
    unbiased_exponent = exponent - bias

    # Recover mantissa
    fraction = 1.0 + mantissa / (1 << mantissa_bits)

    # Reconstruct value
    value = ((-1) ** sign) * fraction * (2 ** unbiased_exponent)

    return value


decoded = decode_custom_fp(
    custom["sign"],
    custom["exponent"],
    custom["mantissa"],
    exponent_bits=7,
    mantissa_bits=8,
)

print(decoded)


original = info["value"]

decoded = decode_custom_fp(
    custom["sign"],
    custom["exponent"],
    custom["mantissa"],
    7,
    8,
)

print(f"Original : {original}")
print(f"Decoded  : {decoded}")
print(f"Error    : {abs(original-decoded)}")