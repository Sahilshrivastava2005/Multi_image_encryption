import hashlib
import numpy as np
import config




def chebyshev_map(p, b):
    # p_{n+1} = cos(b * arccos(p_n))
    return np.cos(b * np.arccos(p))


def logistic_sine_map(x, a, p):
    """
    CICSML lattice update (Eq. (2)):
    x_{n+1}(i) = mod{ (1 - p) * g(x_n(i))
                      + p/2 * ( g(x_n(i+1)) + g(x_n(i-1)) ), 1 }
    g(x) = 4/(a - 0.5) * sin(pi * x)      (Eq. (3))
    [page:2][page:3]
    """
    x = np.asarray(x, dtype=float)
    N = x.size

    # local map g(x)
    g_vals = (4.0 / (a - 0.5)) * np.sin(np.pi * x)

    # periodic neighbors
    idx = np.arange(N)
    left_idx = (idx + 1) % N
    right_idx = (idx - 1) % N

    g_left = g_vals[left_idx]
    g_right = g_vals[right_idx]

    x_next = (1.0 - p) * g_vals + (p / 2.0) * (g_left + g_right)
    x_next = np.mod(x_next, 1.0)
    return x_next


def cicsml_generate(length, a=None, b=None, p0=None, x0=None):
    """
    Generate chaotic sequence using the CICSML system, like CICSMLSystem(Key, M×N). [page:6]

    Here:
    - N_lattice = 9 (number of lattices). [page:6]
    - a, b, p0, x0 come from the key-derivation step (see below).
    - We ignore x0 vector detail here and derive initial x(i) from x0.
    """

    # If a, b, p0, x0 are not passed, use config (or they can be set from key)
    if a is None:
        a = config.A
    if b is None:
        b = config.B
    if p0 is None:
        p0 = config.P0
    if x0 is None:
        x0 = config.X0

    N_lattice = 9  # number of spatiotemporal chaotic system lattices. [page:6]

    # initialize p and x(i) as in paper: x1(i) in (0,1). [page:6]
    p = float(p0)
    # create 9 initial states from x0 (simple spread, paper uses x1(i) from key)
    x = (float(x0) + 0.001 * np.arange(N_lattice)) % 1.0

    seq = []

    # Remove transient effect (T can be set in config, e.g., 100–1000)
    for _ in range(config.TRANSIENT_ITER):
        p = chebyshev_map(p, b)
        x = logistic_sine_map(x, a, p)

    # Generate sequence: flatten time × space
    while len(seq) < length:
        p = chebyshev_map(p, b)
        x = logistic_sine_map(x, a, p)
        seq.extend(x.tolist())

    return np.array(seq[:length])


# ---------------------------------------------------
# KEY → INITIAL CONDITIONS (like authors) [page:6]
# ---------------------------------------------------

def derive_initial_conditions_from_key(user_key):
    """
    Derive (a, b, x0, p0) following Section III-D, Step 1.[page:1]
    """

    # SHA-384 → 48 bytes → 12 segments of 4 bytes (approx. “Key_i”).[page:1]
    key_bytes = hashlib.sha384(user_key.encode()).digest()  # 48 bytes
    assert len(key_bytes) == 48

    # Interpret 12 segments (each 4 bytes) as non‑negative integers Key1..Key12.[page:1]
    Keys = []
    for i in range(12):
        seg = key_bytes[4 * i : 4 * (i + 1)]
        Keys.append(int.from_bytes(seg, byteorder="big", signed=False))

    Key1, Key2 = Keys[0], Keys[1]

    # a = 3.99 + Key1 * 10^{-14} (mod 0.01), b same.[page:1]
    # Implement: take fractional part scaled into [0,0.01), then add 3.99.
    frac1 = (Key1 % 10**14) * 1e-14  # in [0,1)
    frac2 = (Key2 % 10**14) * 1e-14  # in [0,1)

    a = 3.99 + (frac1 * 0.01)        # in [3.99, 4.00)
    b = 3.99 + (frac2 * 0.01)

    # Clamp to (3.57, 4] if you want to stay inside the stated parameter range.[page:1]
    a = min(max(a, 3.57), 4.0)
    b = min(max(b, 3.57), 4.0)

    # x0 from x1(i) = 0.i + Key_{2+i} * 10^{-14}; choose i=1 → Key3.[page:1]
    # 0.1 + tiny offset ensures (0,1).
    x0 = 0.1 + (Keys[2] % 10**14) * 1e-14
    # p0 from p1 = 0.5 + Key12 * 10^{-14}.[page:1]
    p0 = 0.5 + (Keys[11] % 10**14) * 1e-14

    # Strictly avoid hitting 0 or 1 numerically.
    eps = 1e-6
    x0 = min(max(x0, eps), 1.0 - eps)
    p0 = min(max(p0, eps), 1.0 - eps)

    return a, b, x0, p0


def generate_chaos_with_key(user_key, length):
    """
    FINAL FUNCTION USED BY ENCRYPTION & DECRYPTION

    Emulates CICSMLSystem(Key, M×N): [page:6]
    - Build a 384-bit key via SHA-384.
    - Derive (a, b, x0, p0).
    - Run CICSML to get 'length' chaotic values.
    """

    a, b, x0, p0 = derive_initial_conditions_from_key(user_key)

    seq = cicsml_generate(
        length=length,
        a=a,
        b=b,
        p0=p0,
        x0=x0
    )

    return seq


