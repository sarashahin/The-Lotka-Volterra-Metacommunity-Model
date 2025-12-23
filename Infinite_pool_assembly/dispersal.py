############################################
# dispersal.py
############################################
"""
Handles spatial dispersal kernels and invasion pressure.

Logic:
1. If DISPERSAL_KERNEL is provided in config (as a function f(r)):
   - Generates a 2D kernel grid by evaluating f(r).
   - Uses FFT convolution (SciPy on CPU, PyTorch on GPU).
2. Otherwise:
   - Uses Nearest-Neighbor dispersal mixed with Global Mean-Field dispersal.
   - Mixing controlled by LONG_DISTANCE_PROB.
"""
import logging
from accelerator import np, to_cpu, torch # Removed 'device'
import numpy as _real_numpy
import scipy.fft 

from config import (
    NUM_PATCHES_X, NUM_PATCHES_Y, 
    DISPERSAL_RATE, LONG_DISTANCE_PROB
)

# Try to import custom kernel function
try:
    from config import DISPERSAL_KERNEL
except ImportError:
    DISPERSAL_KERNEL = None

logger = logging.getLogger(__name__)

# --- RESTORED CONSTANT ---
# Standard Von Neumann (4-neighbor) Laplacian for local diffusion calculations.
# Used by models to calculate 'away' rates if needed.
LOCAL_DISPERSAL_MATRIX = _real_numpy.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0]
], dtype=_real_numpy.float32)
LOCAL_DISPERSAL_MATRIX /= LOCAL_DISPERSAL_MATRIX.sum()

# Global state
_kernel_fft_torch = None
_kernel_fft_numpy = None
_invasion_pressure = None

def set_invasion_pressure(inv_field):
    global _invasion_pressure
    _invasion_pressure = inv_field

def _precompute_custom_kernel(shape):
    """
    Evaluates the function DISPERSAL_KERNEL(r) on a grid to create
    the 2D convolution kernel, then precomputes its FFT.
    
    NOTE: Dispersal to the origin (r=0) is explicitly removed to ensure
    the kernel represents purely 'away' movement.
    """
    global _kernel_fft_torch, _kernel_fft_numpy
    
    if DISPERSAL_KERNEL is None:
        return

    H, W = shape
    
    # 1. Generate Distance Grid (r)
    y = _real_numpy.arange(H) - H // 2
    x = _real_numpy.arange(W) - W // 2
    xx, yy = _real_numpy.meshgrid(x, y)
    
    # Radial distance r
    r_grid = _real_numpy.sqrt(xx**2 + yy**2)
    
    # 2. Evaluate User Function f(r)
    try:
        kernel_spatial = DISPERSAL_KERNEL(r_grid)
    except Exception:
        kernel_spatial = _real_numpy.vectorize(DISPERSAL_KERNEL)(r_grid)
        
    kernel_spatial = kernel_spatial.astype(_real_numpy.float32)

    # 3. Shift Center for FFT and Zero Origin
    # ifftshift moves the origin (center of grid) to index (0,0)
    kernel_shifted = _real_numpy.fft.ifftshift(kernel_spatial)
    
    # Explicitly remove self-dispersal (staying in same patch)
    kernel_shifted[0, 0] = 0.0

    # 4. Normalize
    # We normalize AFTER zeroing to ensure the total probability of 
    # leaving biomass landing *somewhere else* sums to 1.0.
    k_sum = kernel_shifted.sum()
    if k_sum > 0:
        kernel_shifted /= k_sum
    
    # 5. Compute FFTs
    # A. NumPy/SciPy (CPU)
    _kernel_fft_numpy = scipy.fft.rfft2(kernel_shifted)
    
    # B. PyTorch (GPU)
    if torch is not None:
        # Detect device dynamically
        current_dev = torch.ones(1).device
        k_tensor = torch.from_numpy(kernel_shifted).float().to(current_dev)
        _kernel_fft_torch = torch.fft.rfftn(k_tensor, dim=(-2, -1))
        
    logger.info(f"Custom DISPERSAL_KERNEL(r) evaluated. Shape: {shape}. Origin dispersal removed.")

def _apply_fft_convolution(biomass):
    """
    Applies FFT convolution using the precomputed custom kernel.
    """
    if _kernel_fft_numpy is None:
        _precompute_custom_kernel(biomass.shape[-2:])

    # Branch A: PyTorch
    if torch is not None and torch.is_tensor(biomass):
        if _kernel_fft_torch is None: _precompute_custom_kernel(biomass.shape[-2:])
        
        B_fft = torch.fft.rfftn(biomass, dim=(-2, -1))
        conv_fft = B_fft * _kernel_fft_torch
        return torch.fft.irfftn(conv_fft, s=biomass.shape[-2:], dim=(-2, -1))

    # Branch B: NumPy
    else:
        if not isinstance(biomass, _real_numpy.ndarray): biomass = _real_numpy.asarray(biomass)
        
        B_fft = scipy.fft.rfft2(biomass, axes=(-2, -1))
        conv_fft = B_fft * _kernel_fft_numpy
        return scipy.fft.irfft2(conv_fft, s=biomass.shape[-2:], axes=(-2, -1))

def _apply_nearest_neighbor(biomass):
    """
    Calculates average of 4 nearest neighbors (Von Neumann neighborhood).
    """
    is_torch = torch is not None and torch.is_tensor(biomass)
    
    if is_torch:
        up    = torch.roll(biomass, shifts=1, dims=-2)
        down  = torch.roll(biomass, shifts=-1, dims=-2)
        left  = torch.roll(biomass, shifts=1, dims=-1)
        right = torch.roll(biomass, shifts=-1, dims=-1)
        return 0.25 * (up + down + left + right)
    else:
        up    = _real_numpy.roll(biomass, 1, axis=-2)
        down  = _real_numpy.roll(biomass, -1, axis=-2)
        left  = _real_numpy.roll(biomass, 1, axis=-1)
        right = _real_numpy.roll(biomass, -1, axis=-1)
        return 0.25 * (up + down + left + right)

def compute_dispersal(biomass):
    """
    Calculates the incoming dispersal flux for every patch.
    """
    
    # CASE 1: Custom Kernel Provided -> FFT Convolution
    if DISPERSAL_KERNEL is not None:
        spatial_dist = _apply_fft_convolution(biomass)
    
    # CASE 2: No Kernel -> NN + Global Mixing
    else:
        local_dist = _apply_nearest_neighbor(biomass)
        
        if LONG_DISTANCE_PROB > 0:
            if torch is not None and torch.is_tensor(biomass):
                global_mean = biomass.mean(dim=(-2, -1), keepdim=True)
            else:
                global_mean = biomass.mean(axis=(-2, -1), keepdims=True)
            
            spatial_dist = (1.0 - LONG_DISTANCE_PROB) * local_dist + LONG_DISTANCE_PROB * global_mean
        else:
            spatial_dist = local_dist

    # Apply Rate
    total_input = spatial_dist * DISPERSAL_RATE
    
    if _invasion_pressure is not None:
        total_input += _invasion_pressure
        
    return total_input
