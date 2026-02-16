# analysis.py
import numpy as np
from numpy.linalg import eigvals
from math import log2
from scipy.stats import chisquare
import matplotlib.pyplot as plt


# -------------------------------
# Core CICSML definitions
# -------------------------------

def chebyshev_map(p, b):
    """p_{n+1} = cos(b * arccos(p_n)), |p_n| < 1. [page:2]"""
    return np.cos(b * np.arccos(p))


def local_map_g(x, a):
    """g(x) = 4/(a - 0.5) * sin(pi*x)  (Eq. (2)). [page:2]"""
    return (4.0 / (a - 0.5)) * np.sin(np.pi * x)


def cicsml_step(x, a, p):
    """
    One iteration of CICSML lattice (Eq. (2)):
    x_{n+1}(i) = mod{ (1 - p) * g(x_n(i)) +
                      p/2 * (g(x_n(i+1)) + g(x_n(i-1))), 1 }.
    Periodic boundary (ring). [page:2][page:3]
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    g_vals = local_map_g(x, a)

    idx = np.arange(N)
    left_idx = (idx + 1) % N
    right_idx = (idx - 1) % N

    g_left = g_vals[left_idx]
    g_right = g_vals[right_idx]

    x_next = (1.0 - p) * g_vals + (p / 2.0) * (g_left + g_right)
    x_next = np.mod(x_next, 1.0)
    return x_next


def cml_step(x, a, p):
    """
    Classical CML step (Eq. (1)) for comparison:
    g(y)=a*y*(1-y), fixed p. [page:2]
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    g_vals = a * x * (1 - x)

    idx = np.arange(N)
    left_idx = (idx + 1) % N
    right_idx = (idx - 1) % N

    g_left = g_vals[left_idx]
    g_right = g_vals[right_idx]

    x_next = (1.0 - p) * g_vals + (p / 2.0) * (g_left + g_right)
    return x_next


def cicsml_orbit(a, b, N_lattice=9, n_iter=5000,
                 x0=None, p0=0.5, transient=1000):
    """
    Generate orbit x_n(i) of CICSML. [page:2][page:3]
    Returns:
        xs: shape (n_iter, N_lattice)
        ps: length n_iter
    """
    if x0 is None:
        x0 = np.random.rand(N_lattice)
    x = np.asarray(x0, dtype=float)
    p = float(p0)

    for _ in range(transient):
        p = chebyshev_map(p, b)
        x = cicsml_step(x, a, p)

    xs = np.zeros((n_iter, N_lattice))
    ps = np.zeros(n_iter)
    for n in range(n_iter):
        p = chebyshev_map(p, b)
        x = cicsml_step(x, a, p)
        xs[n] = x
        ps[n] = p
    return xs, ps


def cml_orbit(a, p, N_lattice=9, n_iter=5000,
              x0=None, transient=1000):
    """
    Generate orbit for classical CML for comparison. [page:2][page:3]
    """
    if x0 is None:
        x0 = np.random.rand(N_lattice)
    x = np.asarray(x0, dtype=float)

    for _ in range(transient):
        x = cml_step(x, a, p)

    xs = np.zeros((n_iter, N_lattice))
    for n in range(n_iter):
        x = cml_step(x, a, p)
        xs[n] = x
    return xs


# -------------------------------
# Bifurcation diagram
# -------------------------------

def bifurcation_data_cml(a_values, p=0.5, N_lattice=71,
                         n_transient=2000, n_sample=200):
    """
    Bifurcation data for CML as in Fig. 1(a). [page:3]
    """
    A_list, X_list = [], []
    for a in a_values:
        x = np.random.rand(N_lattice)
        # transient
        for _ in range(n_transient):
            x = cml_step(x, a, p)
        # sample first lattice
        for _ in range(n_sample):
            x = cml_step(x, a, p)
            A_list.append(a)
            X_list.append(x[0])
    return np.array(A_list), np.array(X_list)


def bifurcation_data_cicsml(a_values, b=4.0, p0=0.5,
                            N_lattice=71,
                            n_transient=2000, n_sample=200):
    """
    Bifurcation data for CICSML as in Fig. 1(b). [page:3]
    """
    A_list, X_list = [], []
    for a in a_values:
        x = np.random.rand(N_lattice)
        p = p0
        # transient
        for _ in range(n_transient):
            p = chebyshev_map(p, b)
            x = cicsml_step(x, a, p)
        # sample
        for _ in range(n_sample):
            p = chebyshev_map(p, b)
            x = cicsml_step(x, a, p)
            A_list.append(a)
            X_list.append(x[0])
    return np.array(A_list), np.array(X_list)


def plot_bifurcation():
    a_vals = np.linspace(3.57, 4.0, 300)

    A1, X1 = bifurcation_data_cml(a_vals, p=0.5)
    A2, X2 = bifurcation_data_cicsml(a_vals, b=4.0, p0=0.5)

    plt.figure(figsize=(12, 5))

    # CML
    plt.subplot(1, 2, 1)
    plt.scatter(A1, X1, s=1, color='black')
    plt.title("Bifurcation: CML (p=0.5)")
    plt.xlabel("a")
    plt.ylabel("x")

    # CICSML
    plt.subplot(1, 2, 2)
    plt.scatter(A2, X2, s=1, color='black')
    plt.title("Bifurcation: CICSML (b=4, p1=0.5)")
    plt.xlabel("a")
    plt.ylabel("x")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Spatiotemporal chaos map
# -------------------------------

def spatiotemporal_map_cml(a=3.99, p=0.5, N_lattice=1000, n_iter=1000):
    """
    Spatiotemporal data for CML as in Fig. 2(a). [page:3]
    """
    xs = cml_orbit(a=a, p=p, N_lattice=N_lattice,
                   n_iter=n_iter, transient=1000)
    return xs


def spatiotemporal_map_cicsml(a=3.99, b=4.0, p0=0.5,
                              N_lattice=1000, n_iter=1000):
    """
    Spatiotemporal data for CICSML as in Fig. 2(b). [page:3]
    """
    xs, _ = cicsml_orbit(a=a, b=b, N_lattice=N_lattice,
                         n_iter=n_iter, p0=p0, transient=1000)
    return xs


def plot_spatiotemporal():
    xs_cml = spatiotemporal_map_cml()
    xs_cicsml = spatiotemporal_map_cicsml()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(xs_cml.T, aspect='auto', cmap='jet',
               origin='lower')
    plt.title("Spatiotemporal map: CML (a=3.99)")
    plt.xlabel("time n")
    plt.ylabel("lattice index i")

    plt.subplot(1, 2, 2)
    plt.imshow(xs_cicsml.T, aspect='auto', cmap='jet',
               origin='lower')
    plt.title("Spatiotemporal map: CICSML (a=3.99)")
    plt.xlabel("time n")
    plt.ylabel("lattice index i")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Lyapunov, KED, KEB
# -------------------------------

def jacobian_matrix(x, a, p, use_cicsml=True):
    """
    Jacobi matrix J_k as in Eq. (6), numerical derivative. [page:3]
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    eps = 1e-6
    J = np.zeros((N, N))

    for j in range(N):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps

        if use_cicsml:
            f_plus = cicsml_step(x_plus, a, p)
            f_minus = cicsml_step(x_minus, a, p)
        else:
            f_plus = cml_step(x_plus, a, p)
            f_minus = cml_step(x_minus, a, p)

        J[:, j] = (f_plus - f_minus) / (2 * eps)
    return J


def lyapunov_exponents(a, b_or_p, N_lattice=50,
                       n_iter=200, p0=0.5,
                       use_cicsml=True):
    """
    Approximate Lyapunov exponents λ(i). [page:3]
    """
    x = np.random.rand(N_lattice)
    if use_cicsml:
        b = b_or_p
        p = p0
        for _ in range(200):
            p = chebyshev_map(p, b)
            x = cicsml_step(x, a, p)
    else:
        p = b_or_p
        for _ in range(200):
            x = cml_step(x, a, p)

    R = np.eye(N_lattice)

    for t in range(n_iter):
        if use_cicsml:
            p = chebyshev_map(p, b)
            J = jacobian_matrix(x, a, p, use_cicsml=True)
            x = cicsml_step(x, a, p)
        else:
            J = jacobian_matrix(x, a, p, use_cicsml=False)
            x = cml_step(x, a, p)

        R = J @ R
        if t % 20 == 0:
            Q, R_qr = np.linalg.qr(R)
            R = R_qr

    sigma = eigvals(R)
    lambdas = np.log(np.abs(sigma)) / n_iter
    return np.real(lambdas)


def KED_KEB_from_lyapunov(lambdas):
    """
    KED h and KEB h_u as Eq. (7),(8). [page:3]
    """
    lambdas = np.asarray(lambdas)
    N = len(lambdas)
    if N == 0:
        return 0.0, 0.0
    positive = lambdas[lambdas > 0]
    N_pos = len(positive)
    if N_pos == 0:
        return 0.0, 0.0
    h = np.sum(positive) / N
    hu = N_pos / N
    return h, hu


def plot_KED_KEB():
    """
    KED/KEB vs a for CML and CICSML (like Fig. 3). [page:3]
    """
    a_vals = np.linspace(3.57, 4.0, 20)
    h_cml, hu_cml = [], []
    h_cicsml, hu_cicsml = [], []

    for a in a_vals:
        lamb_cml = lyapunov_exponents(a, 0.5, N_lattice=20,
                                      n_iter=80, use_cicsml=False)
        h1, hu1 = KED_KEB_from_lyapunov(lamb_cml)
        h_cml.append(h1)
        hu_cml.append(hu1)

        lamb_cics = lyapunov_exponents(a, 4.0, N_lattice=20,
                                       n_iter=80, use_cicsml=True)
        h2, hu2 = KED_KEB_from_lyapunov(lamb_cics)
        h_cicsml.append(h2)
        hu_cicsml.append(hu2)

    plt.figure(figsize=(12, 5))

    # KED
    plt.subplot(1, 2, 1)
    plt.plot(a_vals, h_cml, label="CML", marker='o')
    plt.plot(a_vals, h_cicsml, label="CICSML", marker='s')
    plt.title("KED vs a")
    plt.xlabel("a")
    plt.ylabel("h")
    plt.legend()

    # KEB
    plt.subplot(1, 2, 2)
    plt.plot(a_vals, hu_cml, label="CML", marker='o')
    plt.plot(a_vals, hu_cicsml, label="CICSML", marker='s')
    plt.title("KEB vs a")
    plt.xlabel("a")
    plt.ylabel("h_u")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------
# Information entropy
# -------------------------------

def info_entropy(values, n_bins=256):
    """
    Global information entropy of chaotic sequence (Eq. (9),(10)). [page:4]
    """
    v = np.asarray(values)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    idx = np.floor(v * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    H = -np.sum(probs * np.log2(probs))
    H_max = log2(n_bins)
    return H, H_max


def plot_entropy_curve():
    """
    Entropy vs lattice index for CML and CICSML as in Fig. 4. [page:4]
    We take first 100 lattices and compute entropy per lattice.
    """
    a = 3.99
    b = 4.0
    N_lattice = 100
    n_iter = 5000

    xs_cml = cml_orbit(a, p=0.5, N_lattice=N_lattice,
                       n_iter=n_iter, transient=1000)
    xs_cics, _ = cicsml_orbit(a, b, N_lattice=N_lattice,
                              n_iter=n_iter, p0=0.5, transient=1000)

    H_cml = []
    H_cics = []
    for i in range(N_lattice):
        H1, _ = info_entropy(xs_cml[:, i], n_bins=90)
        H2, _ = info_entropy(xs_cics[:, i], n_bins=90)
        H_cml.append(H1)
        H_cics.append(H2)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_lattice + 1), H_cml, label="CML")
    plt.plot(range(1, N_lattice + 1), H_cics, label="CICSML")
    plt.title("Information entropy per lattice (a=3.99)")
    plt.xlabel("Lattice index")
    plt.ylabel("Entropy H")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------
# Simple randomness checks
# -------------------------------

def sequence_to_bits(seq, n_bits=8):
    v = np.asarray(seq)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    ints = np.floor(v * 256).astype(np.uint8)
    bits = np.unpackbits(ints)
    return bits


def nist_frequency_test(bits):
    """
    Monobit frequency test (simple NIST-like). [page:4]
    """
    bits = np.asarray(bits)
    n = len(bits)
    s = np.sum(2 * bits - 1)
    S_obs = abs(s) / np.sqrt(n)
    from math import erfc
    p_value = erfc(S_obs / np.sqrt(2))
    return S_obs, p_value


def chi_square_uniform(seq, n_bins=256):
    """
    Chi-square uniformity test on quantized sequence.
    """
    v = np.asarray(seq)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    idx = np.floor(v * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    expected = np.ones_like(counts) * counts.sum() / n_bins
    chi2, p = chisquare(counts, f_exp=expected)
    return chi2, p


# -------------------------------
# Main: run all analyses & plots
# -------------------------------

if __name__ == "__main__":
    # 1) Bifurcation diagrams (Fig. 1 style)
    plot_bifurcation()

    # 2) Spatiotemporal chaos maps (Fig. 2 style)
    plot_spatiotemporal()

    # 3) KED & KEB curves (Fig. 3 style)
    plot_KED_KEB()

    # 4) Entropy per lattice (Fig. 4 style)
    plot_entropy_curve()

    # 5) Randomness checks (NIST/χ2 idea)
    a = 3.99
    b = 4.0
    xs, _ = cicsml_orbit(a, b, N_lattice=9, n_iter=100100,
                         p0=0.5, transient=100000)
    seq = xs.ravel()

    bits = sequence_to_bits(seq[:100000])
    S_obs, p_freq = nist_frequency_test(bits)
    chi2, p_chi = chi_square_uniform(seq[:100000])

    print("Frequency test S_obs, p:", S_obs, p_freq)
    print("Chi-square uniformity chi2, p:", chi2, p_chi)
