############################################
# dispersal.py
############################################
"""
Dispersal module.
Computes dispersal using a dense GPU-native matrix multiplication.
"""

from accelerator import np
import logging
from config import DISPERSAL_RATE, NUM_PATCHES_X, NUM_PATCHES_Y, LONG_DISTANCE_PROB

# ‹NEW›  –– external invasion tap (uniform per‑patch flux, species‑specific)
_EXTRA_INVASION = None

logger = logging.getLogger(__name__)

def set_invasion_pressure(arr):
    """
    arr: ndarray (S, Ny, Nx) – flux of propagules per patch & species.
         Use zeros to switch the tap off again.
    """
    global _EXTRA_INVASION
    if arr is None:
        _EXTRA_INVASION = None
    else:
        _EXTRA_INVASION = np.asarray(arr, float)

def create_local_dispersal_matrix():
    """
    Create a dense dispersal matrix D (N, N) on the active device.
    N = NUM_PATCHES_X * NUM_PATCHES_Y.
    
    Each patch sends an equal fraction (1/4) of its dispersal flux to each of its 
    4 nearest neighbors (Von Neumann neighborhood) under periodic boundary conditions.
    """
    N = NUM_PATCHES_X * NUM_PATCHES_Y
    D = np.zeros((N, N))

    # Loop logic (runs on CPU to build the matrix, but writes to GPU tensor D)
    for i in range(NUM_PATCHES_Y):
        for j in range(NUM_PATCHES_X):
            idx = i * NUM_PATCHES_X + j
            # 4-neighbor stencil (diagonals excluded)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    if di*dj != 0: continue # Skip diagonals
                    
                    ni = (i + di) % NUM_PATCHES_Y
                    nj = (j + dj) % NUM_PATCHES_X
                    n_idx = ni * NUM_PATCHES_X + nj
                    D[idx, n_idx] = 1.0/4.0
    return DISPERSAL_RATE * D

# Precompute on GPU (if enabled)
LOCAL_DISPERSAL_MATRIX = create_local_dispersal_matrix()

def compute_dispersal(B):
    """
    Compute dispersal flux using dense matrix calc on GPU.
    """
    S = B.shape[0]
    
    # FIX: Use shape instead of .size (which is a method on Tensors)
    total_num_patches = B.shape[1] * B.shape[2]

    # Reshape (S, Y, X) -> (S, N)
    B2 = B.reshape((S, -1))
    
    # Dense Matmul: (D.T @ B2.T).T -> (S, N)
    # LOCAL_DISPERSAL_MATRIX is already on GPU via accelerator.np
    dispersal_flux = (LOCAL_DISPERSAL_MATRIX.T @ B2.T).T

    ## Long-distance flux: sum over patches
    ldd_incoming = LONG_DISTANCE_PROB * dispersal_flux.sum(axis=1) / total_num_patches
        
    incoming_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux + \
        np.broadcast_to(ldd_incoming[:, np.newaxis], dispersal_flux.shape)
        
    if _EXTRA_INVASION is not None:
        incoming_flux += np.asarray(_EXTRA_INVASION).reshape(S, -1)

    return incoming_flux.reshape(B.shape)
