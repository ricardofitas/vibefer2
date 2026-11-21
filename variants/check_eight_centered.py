"""CPU verification of eight 31-bit NTTs and a three-neighbour packed carry.

Run with Python 3, NumPy and gmpy2. Each output square is compared with an
independent GMP Kronecker convolution and with the square of the full integer.
"""

import hashlib
import json
import math
import random
import time
from itertools import islice

import gmpy2
import numpy as np

LENGTH = 1 << 15
ROOT_ORDER = 2 * LENGTH
PRIMES = list(
    islice(
        (
            p
            for k in range((2**31 - 2) // ROOT_ORDER, 0, -1)
            if gmpy2.is_prime(p := k * ROOT_ORDER + 1)
        ),
        8,
    )
)
CRT_MODULUS = math.prod(PRIMES)
BIT_REVERSAL = np.array(
    [int(format(i, "015b")[::-1], 2) for i in range(LENGTH)], dtype=np.int32
)


def powers(root, count, prime):
    values = [1]
    for _ in range(count - 1):
        values.append(values[-1] * root % prime)
    return np.array(values, dtype=np.int64)


def ntt(values, root, prime):
    """Radix-2 NTT in one field. Reduced operands fit a signed 64-bit product."""
    values = values[BIT_REVERSAL].copy()
    size = 2
    while size <= LENGTH:
        blocks = values.reshape(-1, size)
        half = size // 2
        multipliers = powers(pow(root, LENGTH // size, prime), half, prime)
        left = blocks[:, :half].copy()
        right = blocks[:, half:] * multipliers % prime
        blocks[:, :half] = (left + right) % prime
        blocks[:, half:] = (left - right) % prime
        size *= 2
    return values


def negacyclic_square(digits, factor):
    reconstructed = [0] * LENGTH
    modulus = 1
    for prime in PRIMES:
        generator = 2
        while True:
            twist = pow(generator, (prime - 1) // ROOT_ORDER, prime)
            if pow(twist, LENGTH, prime) == prime - 1:
                break
            generator += 1
        root = twist**2 % prime
        values = np.array([digit % prime for digit in digits], dtype=np.int64)
        values = ntt(values * powers(twist, LENGTH, prime) % prime, root, prime)
        values = ntt(values * values % prime, pow(root, -1, prime), prime)
        values = values * powers(pow(twist, -1, prime), LENGTH, prime) % prime
        values = values * (factor * pow(LENGTH, -1, prime) % prime) % prime

        # One exact Garner step lifts the previous coefficients into modulus * prime.
        inverse = pow(modulus, -1, prime)
        reconstructed = [
            old + modulus * (((int(value) - old) * inverse) % prime)
            for old, value in zip(reconstructed, values)
        ]
        modulus *= prime
    return [x if x <= CRT_MODULUS // 2 else x - CRT_MODULUS for x in reconstructed]


def gmp_coefficients(digits, offset, factor):
    """Square as one unsigned GMP integer with 256 bits between coefficients."""
    shifted = [digit + offset for digit in digits]
    width = 32
    assert LENGTH * (2 * offset) ** 2 < 1 << (width * 8)
    packed = gmpy2.mpz.from_bytes(
        b"".join(value.to_bytes(width, "little") for value in shifted), "little"
    )
    raw = memoryview((packed * packed).to_bytes(2 * LENGTH * width, "little"))
    total = sum(shifted)
    prefix = 0
    result = []
    for i, value in enumerate(shifted):
        prefix += value
        low = int.from_bytes(raw[i * width : (i + 1) * width], "little")
        j = i + LENGTH
        high = int.from_bytes(raw[j * width : (j + 1) * width], "little")
        # Subtract the high half at X^L = -1. Remove the cross and constant
        # terms from (shifted - offset)^2 in the same negacyclic ring.
        cross = 2 * prefix - total
        ones = 2 * (i + 1) - LENGTH
        result.append(factor * (low - high - 2 * offset * cross + offset**2 * ones))
    return result


def evaluate(digits, radix):
    """Evaluate a signed positional polynomial using balanced GMP products."""
    layer = list(map(gmpy2.mpz, digits))
    stride = gmpy2.mpz(radix)
    while len(layer) > 1:
        layer = [low + stride * high for low, high in zip(layer[::2], layer[1::2])]
        stride *= stride
    return layer[0]


def centred_divmod(value, radix):
    quotient, unsigned = divmod(value + radix // 2, radix)
    return quotient, unsigned - radix // 2


def preceding(words, i, distance):
    # A carry over the highest position changes sign because B^L = -1.
    j = i - distance
    return words[j] if j >= 0 else -words[j]


def local_carry(coefficients, radix):
    """Re-encode with |digit| <= B/2+1, reading at most three neighbours."""
    low, middle, high = [], [], []
    for value in coefficients:
        quotient, low_digit = centred_divmod(value, radix)
        high_digit, middle_digit = centred_divmod(quotient, radix)
        low.append(low_digit)
        middle.append(middle_digit)
        high.append(high_digit)
    carry, remainder = [], []
    for i in range(LENGTH):
        combined = low[i] + preceding(middle, i, 1) + preceding(high, i, 2)
        quotient, digit = centred_divmod(combined, radix)
        carry.append(quotient)
        remainder.append(digit)

    assert min(carry) >= -1 and max(carry) <= 1
    digits = [remainder[i] + preceding(carry, i, 1) for i in range(LENGTH)]
    bound = radix // 2 + 1
    high_bound = (2 * LENGTH * bound**2 + (radix**2 + radix) // 2) // radix**2
    largest_high = max(map(abs, high))
    largest_digit = max(map(abs, digits))
    assert largest_digit <= bound and largest_high <= high_bound
    return digits, {
        "high_limb_bound": high_bound,
        "observed_max_abs_high": largest_high,
        "carry_counts": {str(k): carry.count(k) for k in (-1, 0, 1)},
        "carry_across_zero": carry[-1],
        "observed_max_abs_word": str(largest_digit),
        "word_bound": str(bound),
    }


def check_squares(base, uniform=False):
    radix = base**4
    bound = radix // 2 + 1
    assert CRT_MODULUS > 4 * LENGTH * bound**2
    if uniform:
        digits = [bound] * LENGTH
    else:
        rng = random.Random(20261121)
        digits = [rng.randint(-bound, bound) for _ in range(LENGTH)]
        digits[:3] = [-bound, bound, -bound]
        digits[-3:] = [bound, -bound, bound]

    modulus = gmpy2.mpz(radix) ** LENGTH + 1
    residue = evaluate(digits, radix) % modulus
    steps = []
    for factor in (2,) if uniform else (2, 1):
        coefficients = negacyclic_square(digits, factor)
        assert coefficients == gmp_coefficients(digits, bound, factor)
        if uniform:
            # Every term in the last coefficient has positive sign.
            assert coefficients[-1] == factor * LENGTH * bound**2
        digits, carry = local_carry(coefficients, radix)
        residue = residue * residue * factor % modulus
        assert evaluate(digits, radix) % modulus == residue
        encoded = residue.to_bytes((residue.bit_length() + 7) // 8, "little")
        steps.append(
            {
                "factor": factor,
                "duplication_bit": factor - 1,
                "carry": carry,
                "coefficient_checks": LENGTH,
                "packed_words": LENGTH,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return {
        "base": base,
        "pattern": "uniform_positive_extreme" if uniform else "dense_boundaries",
        "log2_P": math.log2(CRT_MODULUS),
        "log2_required": math.log2(4 * LENGTH * bound**2),
        "margin": CRT_MODULUS / (4 * LENGTH * bound**2),
        "steps": steps,
    }


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so that all arithmetic checks are active")
    start = time.monotonic()
    # Search in units of two to preserve the even-base requirement exactly.
    low, high = 0, 10**9
    while high - low > 1:
        mid = (low + high) // 2
        bound = (2 * mid) ** 4 // 2 + 1
        if CRT_MODULUS > 4 * LENGTH * bound**2:
            low = mid
        else:
            high = mid

    wraps = []
    radix = 525000000**4
    modulus = gmpy2.mpz(radix) ** LENGTH + 1
    for sign in (-1, 1):
        coefficients = [0] * LENGTH
        coefficients[-1] = sign * (radix // 2 - 1)
        coefficients[-2] = sign * radix * (radix // 2 - 1)
        coefficients[-3] = sign * radix**2
        digits, carry = local_carry(coefficients, radix)
        assert carry["carry_across_zero"] == sign
        assert (
            evaluate(digits, radix) % modulus == evaluate(coefficients, radix) % modulus
        )
        wraps.append(carry)
    tests = [
        check_squares(518715906),
        check_squares(525000000),
        check_squares(525000000, uniform=True),
    ]
    report = {
        "scope": "independent CPU oracle",
        "primes": PRIMES,
        "L": LENGTH,
        "radix": "B=b^4",
        "signed_word_bound": "D=B/2+1",
        "admission_rule": "4*L*D^2 < product(primes)",
        "max_even_base_for_chosen_fields": 2 * low,
        "original_digits_per_residue": 4 * LENGTH,
        "synthetic_carry_wrap_checks": wraps,
        "checked_coefficients": sum(
            step["coefficient_checks"] for case in tests for step in case["steps"]
        ),
        "results": tests,
        "cpu_wall_s": time.monotonic() - start,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
