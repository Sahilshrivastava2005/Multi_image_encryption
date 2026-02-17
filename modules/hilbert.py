import numpy as np


# --------- HILBERT CURVE CORE LOGIC ---------

def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y


def hilbert_index_to_xy(n, d):
    """
    Map 1D Hilbert index d -> 2D (x,y), for an n×n grid (n power of 2).
    """
    t = d
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def generate_hilbert_indices(n):
    """
    Return an array of shape (n*n, 2) giving (x,y) for Hilbert indices 0..n*n-1.
    """
    indices = np.zeros((n * n, 2), dtype=int)
    for i in range(n * n):
        x, y = hilbert_index_to_xy(n, i)
        indices[i] = [x, y]
    return indices


# --------- SCRAMBLING METHODS (PAPER-CONSISTENT) ---------

def hilbert_method1_scramble(A_mat):
    """
    Method 1 (Fig. 7a):
    - A_mat is the matrix of index labels (not pixel values), shape (M,N).
    - We traverse the matrix in Hilbert order and read *the labels*.
    - Output is IC1: a 1D permutation of [0..L-1] in the order specified by
      the Hilbert path applied to A_mat.
    """
    h, w = A_mat.shape
    assert h == w, "Hilbert implementation assumes square matrix (M == N)."

    # 1. Generate Hilbert (x,y) order over the h×h grid
    indices = generate_hilbert_indices(h)  # (h*h, 2): x=row, y=col

    # 2. Extract labels in Hilbert order
    labels_in_hilbert = A_mat[indices[:, 0], indices[:, 1]]

    # 3. Return as 1D permutation (IC1)
    IC1 = labels_in_hilbert.reshape(-1)
    return IC1


def hilbert_method2_scramble(B_mat, fractal_perm):
    """
    Method 2 (Fig. 7b):
    - First arrange the labels B_mat into 1D (row-major).
    - Then assign those 1D labels onto the image grid following Hilbert order.
    - Additionally, we permute by a key (fractal_perm) as in your design.
    - Return IC2: a 1D permutation of [0..L-1].
    """
    h, w = B_mat.shape
    assert h == w, "Hilbert implementation assumes square matrix (M == N)."
    L = h * w

    # 1. Row-major flatten of labels
    flat_labels = B_mat.reshape(-1, order='C')  # 0..L-1 in some scrambled order

    # 2. Hilbert indices for h×h
    hilbert_xy = generate_hilbert_indices(h)  # (L, 2)

    # 3. Optional fractal-based permutation of the 1D labels
    f_flat = fractal_perm.reshape(-1)
    if f_flat.size != L:
        num_repeats = L // f_flat.size + 1
        f_flat = np.tile(f_flat, num_repeats)[:L]
    perm_indices = np.argsort(f_flat)

    flat_labels_perm = flat_labels[perm_indices]

    # 4. Place permuted labels onto the grid in Hilbert order
    grid_labels = np.zeros((h, w), dtype=flat_labels_perm.dtype)
    grid_labels[hilbert_xy[:, 0], hilbert_xy[:, 1]] = flat_labels_perm

    # 5. IC2 is the labels of this grid read back row-major
    IC2 = grid_labels.reshape(-1, order='C')
    return IC2
