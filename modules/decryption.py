import numpy as np
import hashlib
from modules import hilbert
from modules import cicsml
from modules import image_utils

# ==========================================================
# 1️⃣ FRACTAL + INDEX RECONSTRUCTION (Same as Encryption)
# ==========================================================

def build_fractal_matrix(M, N, Keys):

    max_side = max(M, N)
    e = int(np.floor(np.log2(max_side))) + 1
    size = 2 ** e

    FM = np.array([[6, 4],
                   [2, 8]], dtype=np.float64)

    while FM.shape[0] < size:
        Fk = FM
        FM = np.block([
            [2 * Fk, 3 * Fk],
            [4 * Fk, 1 * Fk]
        ])

    FM = FM[:size, :size]

    # ✅ FIX 1 — COLUMN-FIRST FLATTENING
    FM_vec = FM.reshape(-1, order='F')
    FM_vec = FM_vec[:M * N]

    # Fractal permutation
    A = np.argsort(FM_vec)
    A_mat = A.reshape(M, N, order='F')

    # Hilbert scrambles
    A_scrambled, IC1 = hilbert.hilbert_method1_scramble(A_mat)
    B_scrambled, IC2 = hilbert.hilbert_method2_scramble(A_mat, A)

    IC1 = IC1.astype(np.int64).flatten()[:M * N]
    IC2 = IC2.astype(np.int64).flatten()[:M * N]


    # Build F matrix
    size_minus1 = size - 1
    xs, ys = [], []

    for i in range(4):
        xi = (Keys[i] % size_minus1) + 1
        yi = (Keys[i + 4] % size_minus1) + 1
        xs.append(xi - 1)
        ys.append(yi - 1)

    Fmat = np.array([
        [FM[xs[0], ys[0]], FM[xs[1], ys[1]]],
        [FM[xs[2], ys[2]], FM[xs[3], ys[3]]]
    ], dtype=np.int64)

    return FM, IC1, IC2, Fmat



# ==========================================================
# 2️⃣ KEY DERIVATION
# ==========================================================

def derive_key_parts(user_key: str):
    key_bytes = hashlib.sha384(user_key.encode()).digest()
    Keys = []
    for i in range(12):
        seg = key_bytes[4 * i:4 * (i + 1)]
        Keys.append(int.from_bytes(seg, "big", signed=False))
    return Keys


# ==========================================================
# 3️⃣ CHAOS GENERATION
# ==========================================================

def generate_chaos_sequences(user_key, length):
    chaos = cicsml.generate_chaos_with_key(user_key, length=8 * length)

    if isinstance(chaos, tuple) and len(chaos) == 8:
        return chaos

    chaos = np.asarray(chaos).flatten()
    return np.split(chaos, 8)


# ==========================================================
# 4️⃣ CORRECT INVERSE SYNCHRONIZED DISORDER DIFFUSION
# ==========================================================

def inverse_synchronized_disorder_diffusion(CR, CG, CB,
                                            IC1, IC2, Fmat,
                                            D1, D2, D3, D4, D5, D6, D7, D8):
    M, N = CR.shape
    L = M * N

    # Flatten cipher channels
    CRv = CR.reshape(-1, order='F').astype(np.uint8)
    CGv = CG.reshape(-1, order='F').astype(np.uint8)
    CBv = CB.reshape(-1, order='F').astype(np.uint8)

    IC1 = np.asarray(IC1, dtype=np.int64)
    IC2 = np.asarray(IC2, dtype=np.int64)

    # Inverse permutations
    IC1_inv = np.zeros_like(IC1)
    IC1_inv[IC1] = np.arange(L, dtype=np.int64)

    IC2_inv = np.zeros_like(IC2)
    IC2_inv[IC2] = np.arange(L, dtype=np.int64)

    # Chaos to uint8 and correct length
    D1 = D1[:L].astype(np.uint8)
    D2 = D2[:L].astype(np.uint8)
    D3 = D3[:L].astype(np.uint8)
    D4 = D4[:L].astype(np.uint8)
    D5 = D5[:L].astype(np.uint8)
    D6 = D6[:L].astype(np.uint8)
    D7 = D7[:L].astype(np.uint8)
    D8 = D8[:L].astype(np.uint8)

    d7_const = D7[L - 1]
    d8_const = D8[L - 1]

    # ---------- precompute j_all, k_all (same as encryption) ----------
    F = np.asarray(Fmat, dtype=np.int64)
    a, b = F[0]
    c, d = F[1]

    mod_val = max(L - 1, 1)
    idxs = np.arange(L, dtype=np.int64)

    j_all = (a * idxs + b) % mod_val
    k_all = (c * idxs + d) % mod_val

    # ---------- STEP 1: inverse second diffusion (recover TR,TG,TB in IC2 order) ----------
    TRp = np.zeros(L, dtype=np.uint8)
    TGp = np.zeros(L, dtype=np.uint8)
    TBp = np.zeros(L, dtype=np.uint8)

    # Work backwards
    for n in range(L - 1, -1, -1):
        if n == 0:
            TRp[n] = CRv[n] ^ d7_const ^ d8_const ^ D2[0]
            TGp[n] = CGv[n] ^ d7_const ^ d8_const ^ D4[0]
            TBp[n] = CBv[n] ^ d7_const ^ d8_const ^ D6[0]
        elif n == 1:
            TRp[n] = CRv[n] ^ d7_const ^ CRv[n - 1] ^ D2[0]
            TGp[n] = CGv[n] ^ d7_const ^ CGv[n - 1] ^ D4[0]
            TBp[n] = CBv[n] ^ d7_const ^ CBv[n - 1] ^ D6[0]
        else:
            k = k_all[n]
            TRp[n] = CRv[n] ^ CRv[n - 2] ^ CRv[n - 1] ^ D2[k]
            TGp[n] = CGv[n] ^ CGv[n - 2] ^ CGv[n - 1] ^ D4[k]
            TBp[n] = CBv[n] ^ CBv[n - 2] ^ CBv[n - 1] ^ D6[k]

    # Undo IC2 permutation
    TR = TRp[IC2_inv]
    TG = TGp[IC2_inv]
    TB = TBp[IC2_inv]

    # ---------- STEP 2: inverse first diffusion (recover v1,v2,v3 in IC1 order) ----------
    v1p = np.zeros(L, dtype=np.uint8)
    v2p = np.zeros(L, dtype=np.uint8)
    v3p = np.zeros(L, dtype=np.uint8)

    for n in range(L - 1, -1, -1):
        if n == 0:
            v1p[n] = TR[n] ^ d7_const ^ d8_const ^ D1[0]
            v2p[n] = TG[n] ^ d7_const ^ d8_const ^ D3[0]
            v3p[n] = TB[n] ^ d7_const ^ d8_const ^ D5[0]
        elif n == 1:
            v1p[n] = TR[n] ^ d7_const ^ TR[n - 1] ^ D1[0]
            v2p[n] = TG[n] ^ d7_const ^ TG[n - 1] ^ D3[0]
            v3p[n] = TB[n] ^ d7_const ^ TB[n - 1] ^ D5[0]
        else:
            j = n - 1
            # R
            v1p[n] = TR[n] ^ TR[n - 2] ^ TR[n - 1] ^ D1[j]
            # G (inverse of cross‑coupled Eq. 13)
            v2p[n] = TG[n] ^ TR[n - 2] ^ TG[n - 1] ^ D3[j]
            # B
            v3p[n] = TB[n] ^ TB[n - 2] ^ TB[n - 1] ^ D5[j]

    # Undo IC1 permutation
    v1 = v1p[IC1_inv]
    v2 = v2p[IC1_inv]
    v3 = v3p[IC1_inv]

    I1 = v1.reshape(M, N, order='F')
    I2 = v2.reshape(M, N, order='F')
    I3 = v3.reshape(M, N, order='F')

    return I1, I2, I3



# ==========================================================
# 5️⃣ MAIN DECRYPT FUNCTION
# ==========================================================

def decrypt_three_images(C_rgb, user_key,
                         MAP1, MAP2, MAP3):

    H, W, _ = C_rgb.shape

    # Split encrypted channels
    CR = C_rgb[..., 0]
    CG = C_rgb[..., 1]
    CB = C_rgb[..., 2]

    # Regenerate keys
    Keys = derive_key_parts(user_key)

    # Regenerate fractal + IC1, IC2
    FM, IC1, IC2, Fmat = build_fractal_matrix(H, W, Keys)

    # Regenerate chaos
    D1, D2, D3, D4, D5, D6, D7, D8 = \
        generate_chaos_sequences(user_key, H * W)

    # Inverse diffusion
    I1, I2, I3 = inverse_synchronized_disorder_diffusion(
        CR, CG, CB,
        IC1, IC2, Fmat,
        D1, D2, D3, D4, D5, D6, D7, D8
    )

    # Inverse indexed conversion
    P1 = image_utils.indexed_to_rgb(I1, MAP1)
    P2 = image_utils.indexed_to_rgb(I2, MAP2)
    P3 = image_utils.indexed_to_rgb(I3, MAP3)

    return P1, P2, P3
