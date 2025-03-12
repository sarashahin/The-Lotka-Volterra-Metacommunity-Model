############################################
# dispersal.py
############################################
"""
Dispersal module.
Computes the diffusive (Laplacian) flux between patches using periodic boundaries.

Assumptions:
    - The input biomass array B must have shape:
          (NUM_PATCHES_Y, NUM_PATCHES_X, S)
      where S is the number of species.
    - If a spatially variable dispersal field (DISPERSAL_FIELD) is provided in config.py,
      it will be used on a per-patch basis; otherwise, a constant DISPERSAL_RATE is applied uniformly.
    - This implementation uses np.roll to achieve periodic boundaries.
"""

import numpy as np
from config import DISPERSAL_RATE

# Optional:  allow spatial heterogeneity in dispersal,
# define a DISPERSAL_FIELD in  configuration file (shape: (NUM_PATCHES_Y, NUM_PATCHES_X)).
try:
    from config import DISPERSAL_FIELD
except ImportError:
    DISPERSAL_FIELD = None

def compute_dispersal(B):
    """
    Compute the net dispersal flux for a biomass array B.
    
    Parameters:
        B : numpy array of shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)
            Biomass for S species in each patch.
    
    Returns:
        flux : numpy array (same shape as B) of net fluxes computed as:
               flux = D * (B_up + B_down + B_left + B_right - 4 * B)
               
               Where D is:
                 - a spatially variable dispersal field (if DISPERSAL_FIELD is provided), or
                 - the constant DISPERSAL_RATE otherwise.
    """
    # Optional: Check that B has three dimensions.
    if B.ndim != 3:
        raise ValueError("Input array B must have shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)")
        
    # Compute neighbors using np.roll for periodic boundaries.
    B_up    = np.roll(B, shift=1, axis=0)
    B_down  = np.roll(B, shift=-1, axis=0)
    B_left  = np.roll(B, shift=1, axis=1)
    B_right = np.roll(B, shift=-1, axis=1)
    
    # Compute the Laplacian flux.
    net_flux = B_up + B_down + B_left + B_right - 4 * B
    
    # Use spatially variable dispersal rate if provided.
    if DISPERSAL_FIELD is not None:
        # Ensure DISPERSAL_FIELD has shape (NUM_PATCHES_Y, NUM_PATCHES_X).
        # Expand dims to broadcast over species.
        D = np.expand_dims(DISPERSAL_FIELD, axis=-1)
    else:
        D = DISPERSAL_RATE
        
    return D * net_flux
