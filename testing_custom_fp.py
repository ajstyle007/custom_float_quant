from custom_quantizer import decode_fp32, encode_custom_fp, decode_custom_fp

numbers = [
    3.14159265,
    0.123456,
    -8.75,
    100.125,
    0.000123,
]


for x in numbers:

    info = decode_fp32(x)

    custom = encode_custom_fp(
        info["sign"],
        info["unbiased_exponent"],
        info["mantissa"],
        7,
        8,
    )

    y = decode_custom_fp(
        custom["sign"],
        custom["exponent"],
        custom["mantissa"],
        7,
        8,
    )

    print("-"*50)
    print(f"Original : {x}")
    print(f"Decoded  : {y}")
    print(f"Error    : {abs(x-y)}")



# Original : 3.14159265
# Decoded  : 3.140625
# Error    : 0.0009676500000002086
# --------------------------------------------------
# Original : 3.14159265
# Decoded  : 3.140625
# Error    : 0.0009676500000002086
# --------------------------------------------------
# Original : 0.123456
# Decoded  : 0.123291015625
# Error    : 0.0001649843749999963
# --------------------------------------------------
# Original : -8.75
# Decoded  : -8.75
# Error    : 0.0
# --------------------------------------------------
# Original : 100.125
# Decoded  : 100.0
# Error    : 0.125
# --------------------------------------------------
# Original : 0.000123
# Decoded  : 0.00012254714965820312
# Error    : 4.528503417968832e-07