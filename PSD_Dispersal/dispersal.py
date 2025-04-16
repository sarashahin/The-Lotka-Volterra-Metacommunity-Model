############################################
# dispersal.py
############################################
"""
Dispersal module.
Computes dispersal between patches using a source-based approach.
Each patch contributes dispersers to its neighbors based on its own biomass.
Now includes a simple long-distance dispersal component.
A fixed fraction (LONG_DISTANCE_PROB) of the dispersal flux is distributed uniformly
across all patches, representing rare long-distance dispersal events.
This implementation precomputes a dispersal matrix for local dispersal.
"""

import numpy as np
from scipy import sparse
import logging

from config import DISPERSAL_RATE, NUM_PATCHES_X, NUM_PATCHES_Y, LONG_DISTANCE_PROB

logger = logging.getLogger(__name__)

def create_local_dispersal_matrix():
    """
    Create a dispersal matrix D of shape (N, N) where N = NUM_PATCHES_X * NUM_PATCHES_Y,
    representing the contribution from each patch to its 8 neighbors under periodic (wrap-around)
    boundary conditions.
    
    Each patch sends an equal fraction (1/8) of its dispersal flux to each of its eight neighbors.
    """
    N = NUM_PATCHES_X * NUM_PATCHES_Y
    #D = np.zeros((N, N)) ## can be inefficient
    ## The following should give exactly the same results but might be
    ## better for local dispersal:
    # sparse CSC is efficient for multiplication from right
    D = sparse.lil_matrix((N, N))

    # Loop over each patch in grid coordinates.
    for i in range(NUM_PATCHES_Y):
        for j in range(NUM_PATCHES_X):
            idx = i * NUM_PATCHES_X + j  # Linear index for patch (i, j)
            # Loop over the 8 neighboring positions.
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Skip the center cell.
                    if di*dj != 0:
                        continue  # Use only nearest neighbors as conventional.
                    ni = (i + di) % NUM_PATCHES_Y
                    nj = (j + dj) % NUM_PATCHES_X
                    n_idx = ni * NUM_PATCHES_X + nj
                    D[idx, n_idx] = 1.0/4.0
    return DISPERSAL_RATE * D

# Precompute the local dispersal matrix once.
LOCAL_DISPERSAL_MATRIX = create_local_dispersal_matrix()

def compute_dispersal(B):
    """
    Compute the dispersal flux for a biomass array B using a source-based approach,
    including both local (short-distance) and long-distance dispersal.
    
    For each species (each 2D slice in B):
      - Local dispersal is computed based on a biomass excess: patches with biomass higher than the
        average of their eight neighbors contribute dispersers.
      - A fraction LONG_DISTANCE_PROB of the dispersal flux is allocated to long-distance dispersal,
        which is then distributed uniformly across all patches.
    
    This implementation uses a precomputed local dispersal matrix for the local (8-neighbor)
    component.
    
    Parameters:
        B : numpy array of shape (S, NUM_PATCHES_Y, NUM_PATCHES_X)
            Biomass for S species in each patch.
    
    Returns:
        incoming_flux : numpy array (same shape as B) of incoming dispersers to each patch.
    """

    S = B.shape[0]
    total_num_patches = B.size // S

    dispersal_flux = B.reshape((S, -1)) @ LOCAL_DISPERSAL_MATRIX

    ## Long-distance flux:
    ldd_incoming = LONG_DISTANCE_PROB * dispersal_flux.sum(axis=1) / total_num_patches
        
    incoming_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux + \
        np.broadcast_to(ldd_incoming[:, np.newaxis],dispersal_flux.shape)
    
    return incoming_flux.reshape(B.shape)
