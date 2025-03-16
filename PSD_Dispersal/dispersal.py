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
from scipy.sparse import diags, kron, eye
from config import (
    QUANTUM_DISPERSAL,
    DISPERSAL_RATE,
    QUANTUM_DISPERSAL_RATE,
    QUANTUM_PHASE,
)

def create_dispersal_matrix(num_patches_y, num_patches_x):
    """
    Create a sparse dispersal matrix for nearest-neighbor dispersal.

    Parameters:
    - num_patches_y: number of patches in y direction
    - num_patches_x: number of patches in x direction

    Returns:
    - D: dispersal matrix as a sparse matrix
    """
    # Create 1D Laplacian on y and x
    laplacian_1d = diags([1, -2, 1], [-1, 0, 1], shape=(num_patches_y, num_patches_y))

    # Extend to 2D with Kronecker products
    D_y = kron(laplacian_1d, eye(num_patches_x))
    D_x = kron(eye(num_patches_y), laplacian_1d)

    # Combine for 2D Laplacian
    D = D_y + D_x

    return D

def compute_dispersal_matrix(B):
    """
    Compute the classical dispersal flux using the dispersal matrix.

    Parameters:
    - B: input array of shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)

    Returns:
    - flux: computed dispersal flux
    """
    num_patches_y, num_patches_x, _ = B.shape
    D = create_dispersal_matrix(num_patches_y, num_patches_x).tocsc()

    flux = np.zeros_like(B)
    for s in range(B.shape[2]):
        flux[:, :, s] = DISPERSAL_RATE * (B[:, :, s].flatten() @ D).reshape(num_patches_y, num_patches_x)
    
    return flux

def compute_quantum_dispersal_matrix(B):
    """
    Compute the quantum-inspired dispersal flux using the dispersal matrix.

    Parameters:
    - B: input array of shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)

    Returns:
    - flux: computed dispersal flux
    """
    num_patches_y, num_patches_x, _ = B.shape
    D = create_dispersal_matrix(num_patches_y, num_patches_x).tocsc()

    # Create diagonal dispersal matrix
    laplacian_diag_1d = diags([1, 1], [-1, 1], shape=(num_patches_y, num_patches_y))
    D_diag_y = kron(laplacian_diag_1d, laplacian_diag_1d)
    D_diag_x = kron(laplacian_diag_1d, laplacian_diag_1d)
    D_diag = D_diag_y + D_diag_x

    flux = np.zeros_like(B)
    for s in range(B.shape[2]):
        Delta_direct = (B[:, :, s].flatten() @ D).reshape(num_patches_y, num_patches_x)
        Delta_diagonal = (B[:, :, s].flatten() @ D_diag).reshape(num_patches_y, num_patches_x)

        net_flux = (
            np.cos(QUANTUM_PHASE) * Delta_direct
            + np.sin(QUANTUM_PHASE) * Delta_diagonal
        )
        flux[:, :, s] = QUANTUM_DISPERSAL_RATE * net_flux
    
    return flux

def compute_spatial_flux(B):
    """
    Wrapper function to choose between classical or quantum-inspired dispersal,
    depending on the QUANTUM_DISPERSAL flag in config.py.

    Usage in the model:
        flux = compute_spatial_flux(B)

    """
    if QUANTUM_DISPERSAL:
        return compute_quantum_dispersal_matrix(B)
    else:
        return compute_dispersal_matrix(B)
