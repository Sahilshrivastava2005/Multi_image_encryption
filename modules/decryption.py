import numpy as np
from modules import cicsml
from modules import hilbert


# -------- REVERSE DIFFUSION --------

def reverse_diffusion(cipher_img, user_key):
    """
    Reverse the chaotic diffusion process
    """

    h, w = cipher_img.shape
    length = h * w

    chaos_seq = cicsml.generate_chaos_with_key(
        user_key,
        cipher_img,
        length
    )

    chaos_seq = (chaos_seq * 255).astype(np.uint8)

    flat = cipher_img.flatten()

    plain = np.zeros_like(flat)

    # Reverse CBC-style diffusion
    plain[0] = flat[0] ^ chaos_seq[0]

    for i in range(1, length):
        plain[i] = flat[i] ^ chaos_seq[i] ^ flat[i-1]

    plain_img = plain.reshape(h, w)

    return plain_img


def reverse_diffusion_all(D1, D2, D3, user_key):

    print("[INFO] Reversing diffusion...")

    S1 = reverse_diffusion(D1, user_key)
    S2 = reverse_diffusion(D2, user_key)
    S3 = reverse_diffusion(D3, user_key)

    print("[INFO] Diffusion reversed")

    return S1, S2, S3


# -------- REVERSE SCRAMBLING --------

def reverse_permutation(image, perm):

    h, w = image.shape
    flat = image.flatten()

    inv = np.zeros_like(flat)

    inv[perm] = flat

    return inv.reshape(h, w)


def reverse_scramble(scrambled_img, keys):
    """
    Reverse the synchronized scrambling
    """

    chaos_perm = keys["chaos_perm"]
    hilbert_perm = keys["hilbert_perm"]

    # 1. Reverse chaos permutation
    step1 = reverse_permutation(scrambled_img, chaos_perm)

    # 2. Reverse Hilbert permutation
    h, w = step1.shape
    flat = step1.flatten()

    inv = np.zeros_like(flat)

    for i, p in enumerate(hilbert_perm):
        inv[p] = flat[i]

    original = inv.reshape(h, w)

    return original


def reverse_scramble_all(S1, S2, S3, keys1, keys2, keys3):

    print("[INFO] Reversing scrambling...")

    I1 = reverse_scramble(S1, keys1)
    I2 = reverse_scramble(S2, keys2)
    I3 = reverse_scramble(S3, keys3)

    print("[INFO] Scrambling reversed")

    return I1, I2, I3
