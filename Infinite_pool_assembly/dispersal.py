############################################
# dispersal.py
############################################
"""
Dispersal module.
Supports two modes:
1. Direct 3x3 Convolution (Fastest for Local/Nearest-Neighbor)
2. FFT Convolution (Accurate for Long-Distance/Custom Kernels)
Retains legacy LOCAL_DISPERSAL_MATRIX for initialization compatibility.
"""

from accelerator import np
import logging
import config
from config import DISPERSAL_RATE, NUM_PATCHES_X, NUM_PATCHES_Y, LONG_DISTANCE_PROB

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_EXTRA_INVASION = None
_KERNEL_FFT_CACHE = None
_LAST_KERNEL_FUNC = None
_DIRECT_CONV_KERNEL = None

logger = logging.getLogger(__name__)

def set_invasion_pressure(arr):
    global _EXTRA_INVASION
    if arr is None:
        _EXTRA_INVASION = None
    else:
        _EXTRA_INVASION = np.asarray(arr, float)

# --- Legacy Support for Initialization Logic (PSD2/IBM) ---
def create_local_dispersal_matrix():
    """
    Re-added for backward compatibility. 
    Needed by models_psd2.py and models_ibm.py to calculate dispersal_away_rate.
    """
    N = NUM_PATCHES_X * NUM_PATCHES_Y
    D = np.zeros((N, N))
    for i in range(NUM_PATCHES_Y):
        for j in range(NUM_PATCHES_X):
            idx = i * NUM_PATCHES_X + j
            # 4-neighbor stencil
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    if di*dj != 0: continue 
                    ni = (i + di) % NUM_PATCHES_Y
                    nj = (j + dj) % NUM_PATCHES_X
                    n_idx = ni * NUM_PATCHES_X + nj
                    D[idx, n_idx] = 1.0/4.0
    return DISPERSAL_RATE * D

# Defined at module level so other files can import it
LOCAL_DISPERSAL_MATRIX = create_local_dispersal_matrix()


def _get_direct_conv_kernel(device, dtype):
    global _DIRECT_CONV_KERNEL
    if _DIRECT_CONV_KERNEL is None or _DIRECT_CONV_KERNEL.device != device:
        k_dat = torch.tensor([
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]
        ], device=device, dtype=dtype) * DISPERSAL_RATE
        _DIRECT_CONV_KERNEL = k_dat.view(1, 1, 3, 3)
    return _DIRECT_CONV_KERNEL

def _get_spectral_kernel(Ny, Nx, kernel_func, device):
    global _KERNEL_FFT_CACHE, _LAST_KERNEL_FUNC
    if _KERNEL_FFT_CACHE is not None and kernel_func is _LAST_KERNEL_FUNC:
        if hasattr(_KERNEL_FFT_CACHE, 'device') and _KERNEL_FFT_CACHE.device == device:
            return _KERNEL_FFT_CACHE

    logger.info("Generating FFT Dispersal Kernel...")
    y = np.fft.fftfreq(Ny, d=1.0).to(device) * Ny
    x = np.fft.fftfreq(Nx, d=1.0).to(device) * Nx
    Y, X = np.meshgrid(y, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    K = kernel_func(r)
    total = K.sum()
    if total > 0: K /= total
    K_fft = np.fft.rfft2(K)
    _KERNEL_FFT_CACHE = K_fft
    _LAST_KERNEL_FUNC = kernel_func
    return K_fft

def compute_dispersal(B):
    S = B.shape[0]
    total_patches = B.shape[1] * B.shape[2]

    if HAS_TORCH and isinstance(B, torch.Tensor):
        if config.DISPERSAL_KERNEL is not None:
            Ny, Nx = B.shape[1], B.shape[2]
            K_fft = _get_spectral_kernel(Ny, Nx, config.DISPERSAL_KERNEL, B.device)
            B_fft = np.fft.rfft2(B)
            conv_fft = B_fft * K_fft
            dispersal_flux = np.fft.irfft2(conv_fft, s=(Ny, Nx))
            dispersal_flux *= DISPERSAL_RATE
        else:
            b_input = B.unsqueeze(1)
            b_padded = F.pad(b_input, (1, 1, 1, 1), mode='circular')
            k = _get_direct_conv_kernel(B.device, B.dtype)
            dispersal_flux = F.conv2d(b_padded, k).squeeze(1)
    else:
        # Fallback uses the matrix we just restored
        B2 = B.reshape((S, -1))
        dispersal_flux = (LOCAL_DISPERSAL_MATRIX.T @ B2.T).T.reshape(B.shape)

    if LONG_DISTANCE_PROB > 0:
        total_flux_per_species = dispersal_flux.sum(dim=(1, 2)) 
        ldd_incoming = LONG_DISTANCE_PROB * total_flux_per_species / total_patches
        incoming_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux + ldd_incoming.view(S, 1, 1)
    else:
        incoming_flux = dispersal_flux

    if _EXTRA_INVASION is not None:
        if isinstance(B, torch.Tensor):
            if not isinstance(_EXTRA_INVASION, torch.Tensor):
                inv_t = torch.as_tensor(_EXTRA_INVASION, device=B.device, dtype=B.dtype)
            else:
                inv_t = _EXTRA_INVASION
            incoming_flux += inv_t.reshape(B.shape)
        else:
            incoming_flux += np.asarray(_EXTRA_INVASION).reshape(B.shape)

    return incoming_flux
