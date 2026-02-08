import hashlib
import numpy as np
import config


def chebyshev_map(p, b):
    return np.cos(b * np.arccos(p))


def logistic_sine_map(x, a, p):
    return (1 - p) * (a - 0.5) * np.sin(np.pi * x) + \
           p * (a - 0.5) * np.sin(np.pi * x)


def cicsml_generate(length, a=None, b=None, p0=None, x0=None):

    if a is None:
        a = config.A
    if b is None:
        b = config.B
    if p0 is None:
        p0 = config.P0
    if x0 is None:
        x0 = config.X0

    p = p0
    x = x0

    seq = []

    # Remove transient effect
    for _ in range(config.TRANSIENT_ITER):
        p = chebyshev_map(p, b)
        x = logistic_sine_map(x, a, p)
        x = x % 1

    # Generate actual sequence
    for _ in range(length):
        p = chebyshev_map(p, b)
        x = logistic_sine_map(x, a, p)
        x = x % 1
        seq.append(x)

    return np.array(seq)


# ---------------------------------------------------
# NEW DETERMINISTIC KEY SYSTEM (REVERSIBLE)
# ---------------------------------------------------

def derive_initial_conditions_from_key(user_key):
    """
    Derive chaotic initial conditions only from user key
    (NOT from image content)
    """

    key_hash = hashlib.sha256(user_key.encode()).hexdigest()

    part1 = key_hash[:16]
    part2 = key_hash[16:32]

    k1 = int(part1, 16)
    k2 = int(part2, 16)

    x0 = (k1 % 10**8) / 10**8
    p0 = (k2 % 10**8) / 10**8

    x0 = max(x0, 0.000001)
    p0 = max(p0, 0.000001)

    return x0, p0


def generate_chaos_with_key(user_key, length):
    """
    FINAL FUNCTION USED BY ENCRYPTION & DECRYPTION

    Deterministic chaos sequence based ONLY on:
    - user key
    - sequence length
    """

    x0, p0 = derive_initial_conditions_from_key(user_key)

    seq = cicsml_generate(
        length=length,
        a=config.A,
        b=config.B,
        p0=p0,
        x0=x0
    )

    return seq
