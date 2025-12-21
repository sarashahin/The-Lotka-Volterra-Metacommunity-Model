############################################
# dispersal.py
############################################
"""
Dispersal module.
Optimized to use 2D Convolution (O(N)) instead of Dense Matrix Multiplication (O(N^2)).
"""

from accelerator import np
import logging
from config import DISPERSAL_RATE, NUM_PATCHES_X, NUM_PATCHES_Y, LONG_DISTANCE_PROB

# Try importing torch for optimized convolution
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_EXTRA_INVASION = None
_KERNEL_CACHE = None

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
    Legacy dense matrix creator. Kept for CPU fallback or non-toroidal topologies.
    """
    N = NUM_PATCHES_X * NUM_PATCHES_Y
    D = np.zeros((N, N))

    for i in range(NUM_PATCHES_Y):
        for j in range(NUM_PATCHES_X):
            idx = i * NUM_PATCHES_X + j
            # 4-neighbor stencil (diagonals excluded)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    if di*dj != 0: continue 
                    
                    ni = (i + di) % NUM_PATCHES_Y
                    nj = (j + dj) % NUM_PATCHES_X
                    n_idx = ni * NUM_PATCHES_X + nj
                    D[idx, n_idx] = 1.0/4.0
    return DISPERSAL_RATE * D

# Precompute legacy matrix (allocates memory but unused in fast path)
LOCAL_DISPERSAL_MATRIX = create_local_dispersal_matrix()

def compute_dispersal(B):
    """
    Compute dispersal flux.
    Routes to optimized 2D Convolution if running on GPU/Torch.
    """
    global _KERNEL_CACHE
    
    # ---------------------------------------------------------
    # FAST PATH: 2D Convolution (O(N))
    # ---------------------------------------------------------
    if HAS_TORCH and isinstance(B, torch.Tensor):
        S, Y, X = B.shape
        total_num_patches = Y * X

        # 1. Lazy-init the 3x3 kernel on the correct device
        if _KERNEL_CACHE is None or _KERNEL_CACHE.device != B.device:
            # Von Neumann Neighborhood (Up, Down, Left, Right)
            # Each neighbor contributes 0.25 * RATE
            k_dat = torch.tensor([
                [0.0, 0.25, 0.0],
                [0.25, 0.0, 0.25],
                [0.0, 0.25, 0.0]
            ], device=B.device, dtype=B.dtype) * DISPERSAL_RATE
            
            # Reshape for Conv2d: (Out_Channels, In_Channels, H, W)
            # We use 1 channel and apply it to all species effectively via 'batch' dim
            _KERNEL_CACHE = k_dat.view(1, 1, 3, 3)

        # 2. Reshape B to (S, 1, Y, X)
        # We treat Species (S) as the Batch dimension.
        # This allows us to convolve all species simultaneously in parallel.
        b_input = B.unsqueeze(1)

        # 3. Circular Padding (Periodic Boundaries)
        # F.pad format: (Left, Right, Top, Bottom)
        b_padded = F.pad(b_input, (1, 1, 1, 1), mode='circular')

        # 4. Convolve
        flux_4d = F.conv2d(b_padded, _KERNEL_CACHE)
        
        # 5. Extract result (S, Y, X)
        dispersal_flux = flux_4d.squeeze(1)

        # 6. Global / Long Distance Dispersal
        # Sum over spatial dims (axis 1 and 2)
        flux_sum = dispersal_flux.sum(dim=(1, 2)) 
        ldd_incoming = LONG_DISTANCE_PROB * flux_sum / total_num_patches

        # Broadcast LDD back to (S, Y, X)
        # (S,) -> (S, 1, 1) broadcast adds to (S, Y, X)
        incoming_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux + \
                        ldd_incoming.view(S, 1, 1)

        # 7. Add Invasion Pressure
        if _EXTRA_INVASION is not None:
            # Ensure tensor/device match
            if not isinstance(_EXTRA_INVASION, torch.Tensor):
                inv_t = torch.as_tensor(_EXTRA_INVASION, device=B.device, dtype=B.dtype)
            else:
                inv_t = _EXTRA_INVASION
            incoming_flux += inv_t.reshape(B.shape)

        return incoming_flux

    # ---------------------------------------------------------
    # SLOW PATH: Dense Matrix Multiplication (O(N^2))
    # ---------------------------------------------------------
    S = B.shape[0]
    total_num_patches = B.shape[1] * B.shape[2]

    # Reshape (S, Y, X) -> (S, N)
    B2 = B.reshape((S, -1))
    
    # Dense Matmul: (N, N) @ (N, S) -> (N, S)
    dispersal_flux = (LOCAL_DISPERSAL_MATRIX.T @ B2.T).T

    ldd_incoming = LONG_DISTANCE_PROB * dispersal_flux.sum(axis=1) / total_num_patches
        
    incoming_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux + \
        np.broadcast_to(ldd_incoming[:, np.newaxis], dispersal_flux.shape)
        
    if _EXTRA_INVASION is not None:
        incoming_flux += np.asarray(_EXTRA_INVASION).reshape(S, -1)

    return incoming_flux.reshape(B.shape)
