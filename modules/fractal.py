import numpy as np
import config
from modules import hilbert

def generate_base_matrix():
    """
    Initial 2x2 fractal seed matrix as defined in the paper
    """
    return np.array([
        [6, 4],
        [2, 8]
    ])






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

    L = M * N
    FM_vec = FM.reshape(-1, order='F')[:L]

    A = np.argsort(-FM_vec)       

    B = A.copy()

    A_mat = A.reshape(M, N, order='C')
    B_mat = B.reshape(M, N, order='C')

    IC1 = hilbert.hilbert_method1_scramble(A_mat)         
    IC2 = hilbert.hilbert_method2_scramble(B_mat, A_mat)  

    IC1 = np.asarray(IC1, dtype=np.int64)[:L]
    IC2 = np.asarray(IC2, dtype=np.int64)[:L]

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

