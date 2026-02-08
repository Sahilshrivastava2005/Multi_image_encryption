import numpy as np


def inverse_permutation(perm):
    perm = np.array(perm).flatten()
    inv = np.zeros(len(perm), dtype=int)
    inv[perm] = np.arange(len(perm))
    return inv



def reverse_diffusion(cipher_img, user_key):

    from modules import cicsml

    h, w = cipher_img.shape
    length = h * w

    chaos_seq = cicsml.generate_chaos_with_key(user_key, length)
    chaos_seq = (chaos_seq * 255).astype(np.uint8)

    cipher = cipher_img.flatten()
    plain = np.zeros_like(cipher)

    plain[0] = cipher[0] ^ chaos_seq[0]

    for i in range(1, length):
        plain[i] = cipher[i] ^ chaos_seq[i] ^ cipher[i-1]

    return plain.reshape(h, w)


def reverse_diffusion_all(D1, D2, D3, user_key):

    print("[INFO] Reversing diffusion...")

    S1 = reverse_diffusion(D1, user_key)
    S2 = reverse_diffusion(D2, user_key)
    S3 = reverse_diffusion(D3, user_key)

    print("[INFO] Diffusion reversed")

    return S1, S2, S3


def reverse_scramble(scrambled_img, keys):

    h, w = scrambled_img.shape

    chaos_perm = keys["chaos_perm"]
    hilbert_perm = keys["hilbert_perm"]

    flat = scrambled_img.flatten()

    # ---- Reverse chaos permutation ----
    inv_chaos = inverse_permutation(chaos_perm)
    after_chaos = flat[inv_chaos]

    # ---- Reverse hilbert scrambling ----
    # Handle both cases: 1D or coordinate permutation

    if isinstance(hilbert_perm, np.ndarray) and hilbert_perm.ndim == 1:

        inv_hilbert = inverse_permutation(hilbert_perm)
        after_hilbert = after_chaos[inv_hilbert]

    else:
        # Coordinate based mapping
        temp = np.zeros_like(after_chaos)

        for i, coord in enumerate(hilbert_perm):
            r, c = coord
            index = r * w + c
            temp[index] = after_chaos[i]

        after_hilbert = temp

    return after_hilbert.reshape(h, w)



def reverse_scramble_all(S1, S2, S3, k1, k2, k3):

    print("[INFO] Reversing scrambling...")

    I1 = reverse_scramble(S1, k1)
    I2 = reverse_scramble(S2, k2)
    I3 = reverse_scramble(S3, k3)

    print("[INFO] Scrambling reversed")

    return I1, I2, I3
