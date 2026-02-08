import numpy as np
from modules import hilbert
from modules import fractal
from modules import cicsml


def chaotic_permutation(chaos_seq):
    return np.argsort(chaos_seq)


def synchronized_scramble(image, user_key):

    h, w = image.shape
    length = h * w

    # Hilbert scrambling
    hilbert_scrambled, hilbert_perm = hilbert.hilbert_method1_scramble(image)

    # Generate deterministic chaos sequence
    chaos_seq = cicsml.generate_chaos_with_key(user_key, length)

    chaos_perm = chaotic_permutation(chaos_seq)

    flat = hilbert_scrambled.flatten()

    scrambled = flat[chaos_perm]

    scrambled = scrambled.reshape(h, w)

    return scrambled, {
        "hilbert_perm": hilbert_perm,
        "chaos_perm": chaos_perm
    }


def scramble_all_images(I1, I2, I3, user_key):

    print("[INFO] Performing synchronized scrambling...")

    S1, keys1 = synchronized_scramble(I1, user_key)
    S2, keys2 = synchronized_scramble(I2, user_key)
    S3, keys3 = synchronized_scramble(I3, user_key)

    print("[INFO] Scrambling completed")

    return (S1, keys1), (S2, keys2), (S3, keys3)


def diffuse_image(scrambled_img, user_key):

    h, w = scrambled_img.shape
    length = h * w

    chaos_seq = cicsml.generate_chaos_with_key(user_key, length)
    chaos_seq = (chaos_seq * 255).astype(np.uint8)

    flat = scrambled_img.flatten()
    cipher = np.zeros_like(flat)

    cipher[0] = flat[0] ^ chaos_seq[0]

    for i in range(1, length):
        cipher[i] = flat[i] ^ chaos_seq[i] ^ cipher[i-1]

    return cipher.reshape(h, w)


def diffuse_all_images(S1, S2, S3, user_key):

    print("[INFO] Performing synchronized diffusion...")

    D1 = diffuse_image(S1, user_key)
    D2 = diffuse_image(S2, user_key)
    D3 = diffuse_image(S3, user_key)

    print("[INFO] Diffusion completed")

    return D1, D2, D3
