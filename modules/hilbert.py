import numpy as np
import config

# --------- HILBERT CURVE CORE LOGIC ---------

def rot(n, x, y, rx, ry):
    """
    Rotation helper for Hilbert curve to maintain continuity.
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y

def hilbert_index_to_xy(n, d):
    """
    Convert a 1D Hilbert index 'd' back into 2D (x, y) coordinates.
    n must be a power of 2.
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
    Generates a lookup table of (x, y) coordinates following the Hilbert path.
    Essential for Method 2 mapping.
    """
    indices = np.zeros((n * n, 2), dtype=int)
    for i in range(n * n):
        x, y = hilbert_index_to_xy(n, i)
        indices[i] = [x, y]
    return indices

# --------- SCRAMBLING METHODS ---------

def hilbert_method1_scramble(image):
    """
    Method 1: Direct Hilbert scanning. 
    Reads the image pixels in the order they appear on the Hilbert curve.
    """
    h, w = image.shape
    indices = generate_hilbert_indices(h)
    
    # Extract pixels in Hilbert order
    scrambled_flat = image[indices[:, 0], indices[:, 1]]
    
    return scrambled_flat.reshape(h, w), indices

def inverse_method1(scrambled, indices):
    """
    Reverse Method 1: Place pixels from 1D back to their 2D Hilbert coordinates.
    """
    h, w = scrambled.shape
    flat = scrambled.flatten()
    
    reconstructed = np.zeros((h, w), dtype=scrambled.dtype)
    reconstructed[indices[:, 0], indices[:, 1]] = flat
    
    return reconstructed

def hilbert_method2_scramble(indexed_img, fractal_perm):
    """
    Method 2: Hybrid Hilbert-Fractal scrambling.
    Uses fractal values to re-order the Hilbert-scanned pixels.
    """
    h, w = indexed_img.shape
    indices = generate_hilbert_indices(h)
    
    # 1. Hilbert Scan (2D -> 1D)
    hilbert_flat = indexed_img[indices[:, 0], indices[:, 1]]
    
    # 2. Prepare Fractal Key (Tile if smaller than image)
    f_flat = fractal_perm.flatten()
    if f_flat.size != hilbert_flat.size:
        num_repeats = (hilbert_flat.size // f_flat.size) + 1
        f_flat = np.tile(f_flat, num_repeats)[:hilbert_flat.size]
    
    # 3. Create permutation based on fractal values
    perm_indices = np.argsort(f_flat)
    
    # 4. Scramble the Hilbert sequence
    scrambled_flat = hilbert_flat[perm_indices]
    
    return scrambled_flat.reshape(h, w), perm_indices

def inverse_method2(scrambled, perm_indices):
    """
    Reverse Method 2: Reverse fractal permutation, then reverse Hilbert scan.
    """
    h, w = scrambled.shape
    indices = generate_hilbert_indices(h)
    flat = scrambled.flatten()
    
    # 1. Reverse the fractal permutation
    inv_perm = np.argsort(perm_indices)
    hilbert_flat = flat[inv_perm]
    
    # 2. Map 1D Hilbert sequence back to 2D
    original = np.zeros((h, w), dtype=scrambled.dtype)
    original[indices[:, 0], indices[:, 1]] = hilbert_flat
    
    return original