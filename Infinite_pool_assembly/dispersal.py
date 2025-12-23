############################################
# dispersal.py
############################################
"""
Handles spatial dispersal kernels and invasion pressure.
Fully encapsulated using 'accelerator' for backend agnostic FFT and rolling.
"""
import logging
from accelerator import np, to_cpu, fft, torch # torch can be None
import numpy as _real_numpy 

from config import (
    NUM_PATCHES_X, NUM_PATCHES_Y, 
    DISPERSAL_RATE, LONG_DISTANCE_PROB
)

try:
    from config import DISPERSAL_KERNEL
except ImportError:
    DISPERSAL_KERNEL = None

logger = logging.getLogger(__name__)

LOCAL_DISPERSAL_MATRIX = _real_numpy.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0]
], dtype=_real_numpy.float32)
LOCAL_DISPERSAL_MATRIX /= LOCAL_DISPERSAL_MATRIX.sum()

# Global state
_kernel_fft = None 
_invasion_pressure = None

def set_invasion_pressure(inv_field):
    global _invasion_pressure
    _invasion_pressure = inv_field

def _precompute_custom_kernel(shape):
    """
    Evaluates DISPERSAL_KERNEL(r), normalizes it, and precomputes the FFT.
    Uses 'accelerator.fft' to handle CPU/GPU logic automatically.
    """
    global _kernel_fft
    
    if DISPERSAL_KERNEL is None:
        return

    H, W = shape
    
    # 1. Generate Distance Grid (r) on CPU first
    y = _real_numpy.arange(H) - H // 2
    x = _real_numpy.arange(W) - W // 2
    xx, yy = _real_numpy.meshgrid(x, y)
    r_grid = _real_numpy.sqrt(xx**2 + yy**2)
    
    # 2. Evaluate User Function f(r)
    try:
        kernel_spatial = DISPERSAL_KERNEL(r_grid)
    except Exception:
        kernel_spatial = _real_numpy.vectorize(DISPERSAL_KERNEL)(r_grid)
        
    kernel_spatial = kernel_spatial.astype(_real_numpy.float32)

    # 3. Shift Center and Zero Origin
    kernel_shifted = _real_numpy.fft.ifftshift(kernel_spatial)
    kernel_shifted[0, 0] = 0.0

    # 4. Normalize
    k_sum = kernel_shifted.sum()
    if k_sum > 0: kernel_shifted /= k_sum
    
    # 5. Move to Accelerator and Compute FFT
    # accelerator.np.array handles moving to GPU if enabled
    k_device = np.array(kernel_shifted)
    
    # Use unified FFT wrapper
    _kernel_fft = fft.rfft2(k_device)
        
    logger.info(f"Custom DISPERSAL_KERNEL(r) evaluated. Shape: {shape}. Origin dispersal removed.")

def _apply_fft_convolution(biomass):
    """
    Applies FFT convolution using the precomputed custom kernel.
    """
    if _kernel_fft is None:
        _precompute_custom_kernel(biomass.shape[-2:])

    # 1. Forward FFT
    # axes=(-2, -1) is implied for 2D FFTs on last dims
    B_fft = fft.rfft2(biomass)
    
    # 2. Convolution (Element-wise multiplication in Freq Domain)
    conv_fft = B_fft * _kernel_fft
    
    # 3. Inverse FFT
    return fft.irfft2(conv_fft, s=biomass.shape[-2:])

def _apply_nearest_neighbor(biomass):
    """
    Calculates average of 4 nearest neighbors (Von Neumann).
    Uses accelerator.np.roll which maps to torch.roll or numpy.roll.
    """
    up    = np.roll(biomass, 1, axis=-2)
    down  = np.roll(biomass, -1, axis=-2)
    left  = np.roll(biomass, 1, axis=-1)
    right = np.roll(biomass, -1, axis=-1)
    return 0.25 * (up + down + left + right)

def compute_dispersal(biomass):
    """
    Calculates the incoming dispersal flux for every patch.
    """
    # CASE 1: Custom Kernel -> FFT
    if DISPERSAL_KERNEL is not None:
        spatial_dist = _apply_fft_convolution(biomass)
    
    # CASE 2: NN + Global Mixing
    else:
        local_dist = _apply_nearest_neighbor(biomass)
        
        if LONG_DISTANCE_PROB > 0:
            # accelerator.np.mean handles dim/axis differences
            global_mean = np.mean(biomass, axis=(-2, -1), keepdims=True)
            spatial_dist = (1.0 - LONG_DISTANCE_PROB) * local_dist + LONG_DISTANCE_PROB * global_mean
        else:
            spatial_dist = local_dist

    # Apply Rate
    total_input = spatial_dist * DISPERSAL_RATE
    
    if _invasion_pressure is not None:
        total_input += _invasion_pressure
        
    return total_input
