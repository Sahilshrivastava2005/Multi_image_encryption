import hashlib
import numpy as np
import config


def chebyshev_map(p, b):
    """
    Chebyshev chaotic component
    T_b(p) = cos(b * arccos(p))
    """
    return np.cos(b * np.arccos(p))


def logistic_sine_map(x, a, p):
    """
    Combined Logistic-Sine chaotic map
    """
    return (1 - p) * (a - 0.5) * np.sin(np.pi * x) + \
           p * (a - 0.5) * np.sin(np.pi * x)


def cicsml_generate(length, a=None, b=None, p0=None, x0=None):
    """
    Generate chaotic sequence using CICSML system

    Parameters taken from config if not provided
    """

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


import hashlib


def image_based_key(image):
    """
    Generate initial chaotic parameters from image content
    """

    h = hashlib.sha256()

    h.update(image.tobytes())

    digest = h.hexdigest()

    # Convert hash to numeric initial values
    key_int = int(digest[:16], 16)

    x0 = (key_int % 100000) / 100000
    p0 = ((key_int >> 16) % 100000) / 100000

    return x0, p0

def generate_image_chaos(image_shape):
    """
    Generate chaotic sequence for full image size
    """

    h, w = image_shape
    length = h * w

    return cicsml_generate(length)


def combine_keys(user_key: str, image):
    """
    Combine user secret key with image-based key
    to generate final initial conditions
    """

    # Hash user key
    h1 = hashlib.sha256(user_key.encode()).hexdigest()

    # Hash image content
    h2 = hashlib.sha256(image.tobytes()).hexdigest()

    # Combine both hashes
    final_hash = hashlib.sha256((h1 + h2).encode()).hexdigest()

    return final_hash


def derive_initial_conditions(final_hash):
    """
    Convert final hash to chaotic initial values
    """

    # Split hash into parts
    part1 = final_hash[:16]
    part2 = final_hash[16:32]

    k1 = int(part1, 16)
    k2 = int(part2, 16)

    # Normalize to (0,1)
    x0 = (k1 % 10**8) / 10**8
    p0 = (k2 % 10**8) / 10**8

    # Ensure not zero
    x0 = max(x0, 0.000001)
    p0 = max(p0, 0.000001)

    return x0, p0

def generate_final_chaotic_keys(user_key, image):
    """
    Main function to generate final chaotic parameters
    """

    print("[INFO] Generating integrated secret keys...")

    final_hash = combine_keys(user_key, image)

    x0, p0 = derive_initial_conditions(final_hash)

    print("[INFO] Final initial conditions generated")

    return x0, p0

def generate_chaos_with_key(user_key, image, length):
    """
    Generate chaotic sequence using integrated key system
    """

    x0, p0 = generate_final_chaotic_keys(user_key, image)

    seq = cicsml_generate(
        length=length,
        a=config.A,
        b=config.B,
        p0=p0,
        x0=x0
    )

    return seq
