import numpy as np
import config


def generate_base_matrix():
    """
    Initial 2x2 fractal seed matrix as defined in the paper
    """
    return np.array([
        [6, 4],
        [2, 8]
    ])


def expand_fractal_matrix(FM, order):
    """
    Iteratively expand fractal matrix to required order
    Following equation (8) from the paper
    """

    for i in range(2, order + 1):

        size = FM.shape[0]
        E = np.ones_like(FM)

        factor = (2 ** (2 * (i - 1)))

        B1 = (FM[0, 0] - 1) * factor * E + FM
        B2 = (FM[0, 1] - 1) * factor * E + FM
        B3 = (FM[1, 0] - 1) * factor * E + FM
        B4 = (FM[1, 1] - 1) * factor * E + FM

        # Combine blocks to form next order matrix
        FM = np.block([
            [B1, B2],
            [B3, B4]
        ])

    return FM


def generate_fractal_matrix():
    """
    Main function to generate fractal matrix using order from config
    """

    print("[INFO] Generating fractal matrix...")

    base = generate_base_matrix()

    FM = expand_fractal_matrix(base, config.FRACTAL_ORDER)

    print("[INFO] Fractal matrix generated with size:", FM.shape)

    return FM


def get_fractal_permutation(FM):
    """
    Convert fractal matrix into permutation vector
    Used later for scrambling
    """

    flat = FM.flatten()

    # Get permutation order based on sorted indices
    perm = np.argsort(flat)

    return perm
