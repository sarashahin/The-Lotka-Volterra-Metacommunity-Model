
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
    D = np.zeros((N, N))
    
    # Loop over each patch in grid coordinates.
    for i in range(NUM_PATCHES_Y):
        for j in range(NUM_PATCHES_X):
            idx = i * NUM_PATCHES_X + j  # Linear index for patch (i, j)
            # Loop over the 8 neighboring positions.
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Skip the center cell.
                    ni = (i + di) % NUM_PATCHES_Y
                    nj = (j + dj) % NUM_PATCHES_X
                    n_idx = ni * NUM_PATCHES_X + nj
                    D[idx, n_idx] = 1 / 8.0
    return D

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
        outgoing_flux : numpy array (same shape as B) of outgoing dispersers from each patch.
        incoming_flux : numpy array (same shape as B) of incoming dispersers to each patch.
    """
    if B.ndim != 3:
        raise ValueError("Input array B must have shape (S, NUM_PATCHES_Y, NUM_PATCHES_X)")
    
    # total_num_patches = NUM_PATCHES_X * NUM_PATCHES_Y
    # S = B.shape[0]
    
    # outgoing_flux = np.zeros_like(B)
    # incoming_flux = np.zeros_like(B)
    
    # Process each species separately.
    # for s in range(S):
    #     species_B = B[s]
        
    #     # Compute the average neighbor biomass using the precomputed matrix.
    #     # Flatten species_B to a vector.
    #     species_B_flat = species_B.flatten()
    #     neighbor_mean_flat = LOCAL_DISPERSAL_MATRIX.dot(species_B_flat)
    #     neighbor_mean = neighbor_mean_flat.reshape(species_B.shape)
        
        
        # Compute the biomass difference (source minus neighbor mean).
        # biomass_diff = species_B - neighbor_mean

        # # Only allow dispersal from patches where the biomass exceeds the neighbor average.
        # dispersal_mask = biomass_diff > 0

        # # Compute the dispersal flux (proportional to biomass and its excess) for each patch.
        # dispersal_flux = DISPERSAL_RATE * species_B * biomass_diff / (species_B + 1e-10)
        # dispersal_flux[~dispersal_mask] = 0

        
        # Passive dispersal: each patch disperses a fixed fraction of its biomass,
        # regardless of local biomass differences.
    #     dispersal_flux = DISPERSAL_RATE * species_B

        
    #     # Split the flux into local and long-distance components.
    #     local_flux = (1 - LONG_DISTANCE_PROB) * dispersal_flux
    #     ldd_flux = LONG_DISTANCE_PROB * dispersal_flux
        
    #     # Record the total outgoing flux.
    #     outgoing_flux[s] = dispersal_flux
        
    #     # Compute local incoming flux via matrix multiplication:
    #     local_flux_flat = local_flux.flatten()
    #     local_incoming_flat = LOCAL_DISPERSAL_MATRIX.T.dot(local_flux_flat)
    #     local_incoming = local_incoming_flat.reshape(species_B.shape)
        
    #     # Compute long-distance incoming flux:
    #     # Sum the long-distance flux from all patches and distribute uniformly.
    #     total_ldd_flux = np.sum(ldd_flux)
    #     ldd_incoming = total_ldd_flux / total_num_patches
        
    #     # Total incoming flux is the sum of local and long-distance contributions.
    #     incoming_flux[s] = local_incoming + ldd_incoming
    
    # return outgoing_flux, incoming_flux
    
    
    S, ny, nx = B.shape
    N = nx * ny

    # Outgoing flux is proportional to biomass.
    outgoing_flux = DISPERSAL_RATE * B

    # Split outgoing flux.
    local_flux = (1 - LONG_DISTANCE_PROB) * outgoing_flux
    ldd_flux   = LONG_DISTANCE_PROB * outgoing_flux

    # Compute local incoming flux (vectorized over all species).
    local_incoming = (local_flux.reshape(S, -1) @ LOCAL_DISPERSAL_MATRIX.T).reshape(B.shape)
    
    # Compute long-distance flux: distribute uniformly.
    total_ldd_flux = np.sum(ldd_flux, axis=(1, 2), keepdims=True)
    ldd_incoming = total_ldd_flux / N

    incoming_flux = local_incoming + ldd_incoming

    return outgoing_flux, incoming_flux
