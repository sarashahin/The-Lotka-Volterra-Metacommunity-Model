############################################
# dispersal.py
############################################
"""
Dispersal module with both classical and quantum-inspired options.

- Classical:
    flux = DISPERSAL_RATE * (B_up + B_down + B_left + B_right - 4*B)

- Quantum-inspired:
    flux = QUANTUM_DISPERSAL_RATE * [cos(phi)*Delta_direct(B) + sin(phi)*Delta_diagonal(B)]

  where:
    Delta_direct(B)   = B_up + B_down + B_left + B_right - 4*B
    Delta_diagonal(B) = B_up_left + B_up_right + B_down_left + B_down_right - 4*B

Call `compute_spatial_flux(B)` from the model; it will automatically pick
the correct function (classical vs. quantum-inspired) based on the
QUANTUM_DISPERSAL flag in config.py.
"""

import numpy as np
from config import (
    QUANTUM_DISPERSAL,
    DISPERSAL_RATE,
    QUANTUM_DISPERSAL_RATE,
    QUANTUM_PHASE,
)

def compute_dispersal(B):
    """
    Classical dispersal flux using the 4-neighbor Laplacian.
    
    flux = DISPERSAL_RATE * (B_up + B_down + B_left + B_right - 4*B)
    """
    if B.ndim != 3:
        raise ValueError("Input array B must have shape (NUM_PATCHES_Y, NUM_PATCHES_X, S).")

    # Periodic boundary neighbors:
    B_up    = np.roll(B, shift=1, axis=0)
    B_down  = np.roll(B, shift=-1, axis=0)
    B_left  = np.roll(B, shift=1, axis=1)
    B_right = np.roll(B, shift=-1, axis=1)
    
    net_flux = B_up + B_down + B_left + B_right - 4.0 * B
    return DISPERSAL_RATE * net_flux

def compute_quantum_dispersal(B):
    """
    Quantum-inspired dispersal flux.

    dB/dt += QUANTUM_DISPERSAL_RATE * [
        cos(QUANTUM_PHASE)*Delta_direct(B) + sin(QUANTUM_PHASE)*Delta_diagonal(B)
    ]

    where:
      Delta_direct(B)   = B_up + B_down + B_left + B_right - 4*B
      Delta_diagonal(B) = B_up_left + B_up_right + B_down_left + B_down_right - 4*B

    Periodic boundaries are assumed via np.roll.
    """
    if B.ndim != 3:
        raise ValueError("Input array B must have shape (NUM_PATCHES_Y, NUM_PATCHES_X, S).")
    
    # Direct neighbors (4-neighbor)
    B_up    = np.roll(B, shift=1, axis=0)
    B_down  = np.roll(B, shift=-1, axis=0)
    B_left  = np.roll(B, shift=1, axis=1)
    B_right = np.roll(B, shift=-1, axis=1)
    Delta_direct = B_up + B_down + B_left + B_right - 4.0 * B

    # Diagonal neighbors (4-diagonal)
    B_up_left    = np.roll(np.roll(B, shift=1, axis=0),  shift=1,  axis=1)
    B_up_right   = np.roll(np.roll(B, shift=1, axis=0),  shift=-1, axis=1)
    B_down_left  = np.roll(np.roll(B, shift=-1, axis=0), shift=1,  axis=1)
    B_down_right = np.roll(np.roll(B, shift=-1, axis=0), shift=-1, axis=1)
    Delta_diagonal = (B_up_left + B_up_right + B_down_left + B_down_right) - 4.0 * B

    # Combine direct + diagonal with phase-based interference
    net_flux = (
        np.cos(QUANTUM_PHASE) * Delta_direct
        + np.sin(QUANTUM_PHASE) * Delta_diagonal
    )
    return QUANTUM_DISPERSAL_RATE * net_flux

def compute_spatial_flux(B):
    """
    Wrapper function to choose between classical or quantum-inspired dispersal,
    depending on the QUANTUM_DISPERSAL flag in config.py.

    Usage in the model:
        flux = compute_spatial_flux(B)

    """
    if QUANTUM_DISPERSAL:
        return compute_quantum_dispersal(B)
    else:
        return compute_dispersal(B)
