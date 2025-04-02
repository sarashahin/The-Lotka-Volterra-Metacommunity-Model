############################################
# dispersal.py
############################################
"""
Dispersal module with multiple dispersal kernels.
Implements exponential, Gaussian, and Lévy flight dispersal kernels.
Includes environmental effects and seed size effects.
Based on Bullock et al. (2005) - Long-distance seed dispersal.
"""

import numpy as np
import logging
from config_base import IBMConfig


logger = logging.getLogger(__name__)

def generate_habitat_quality(config=None):
    """
    Generate spatially heterogeneous habitat quality.
    """
    if config is None:
        config = IBMConfig()
        
    # Create coordinate arrays
    y_coords, x_coords = np.meshgrid(np.arange(config.NUM_PATCHES_Y), np.arange(config.NUM_PATCHES_X))
    
    # Generate random field using 2D Fourier transform
    kx = 2 * np.pi * np.fft.fftfreq(config.NUM_PATCHES_X)
    ky = 2 * np.pi * np.fft.fftfreq(config.NUM_PATCHES_Y)
    kxx, kyy = np.meshgrid(kx, ky)
    k = np.sqrt(kxx**2 + kyy**2)
    
    # Power spectrum
    power = np.exp(-k**2 * config.HABITAT_QUALITY['scale']**2)
    
    # Generate random field
    phase = np.random.rand(*power.shape) * 2 * np.pi
    field = np.fft.ifft2(np.sqrt(power) * np.exp(1j * phase)).real
    
    # Normalize and scale
    field = (field - field.min()) / (field.max() - field.min())
    habitat = config.HABITAT_QUALITY['base'] + config.HABITAT_QUALITY['variation'] * (field - 0.5)
    
    return habitat

def calculate_wind_effect(distances, angles, config=None):
    """
    Calculate wind effect on dispersal.
    """
    if config is None:
        config = IBMConfig()
        
    # Calculate angle between wind and dispersal direction
    angle_diff = angles - config.WIND_DIRECTION
    
    # Wind effect (positive means wind-aided dispersal)
    wind_effect = np.cos(angle_diff) * config.WIND_STRENGTH
    
    return wind_effect

def calculate_seed_size_effects(seed_size, config=None):
    """
    Calculate effects of seed size on dispersal and establishment.
    """
    if config is None:
        config = IBMConfig()
        
    # Effect on dispersal distance
    dispersal_factor = (config.SEED_SIZE_EFFECTS['dispersal_distance']['base'] + 
                       config.SEED_SIZE_EFFECTS['dispersal_distance']['effect'] * seed_size)
    
    # Effect on establishment probability
    establishment_factor = (config.SEED_SIZE_EFFECTS['establishment']['base'] + 
                          config.SEED_SIZE_EFFECTS['establishment']['effect'] * seed_size)
    
    return dispersal_factor, establishment_factor

def calculate_distance_matrix(config=None):
    """
    Calculate the distance matrix between all patches using optimized vectorized operations.
    """
    if config is None:
        config = IBMConfig()
        
    # Create coordinate arrays
    y_coords, x_coords = np.meshgrid(np.arange(config.NUM_PATCHES_Y), np.arange(config.NUM_PATCHES_X))
    
    # Reshape to 2D array of coordinates
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))
    
    # Vectorized distance calculation using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Calculate angles for wind effect using vectorized operations
    angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])
    
    # Set diagonal to zero (no self-dispersal)
    np.fill_diagonal(distances, np.inf)
    
    return distances, angles

def dispersal_kernel_vectorized(distances, angles, kernel_type, params, seed_size, config=None):
    """
    Vectorized dispersal kernel calculation with environmental and seed size effects.
    """
    if config is None:
        config = IBMConfig()
        
    # Base dispersal probability
    if kernel_type == 'exponential':
        probs = np.exp(-distances / params['lambda'])
    elif kernel_type == 'gaussian':
        probs = np.exp(-distances**2 / (2 * params['sigma']**2))
    elif kernel_type == 'levy':
        mask = distances <= params['cutoff']
        probs = np.zeros_like(distances)
        probs[mask] = np.exp(-params['alpha'] * np.log(distances[mask] + 1))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Apply seed size effects
    dispersal_factor, _ = calculate_seed_size_effects(seed_size, config)
    probs = probs * dispersal_factor
    
    # Apply wind effect
    wind_effect = calculate_wind_effect(distances, angles, config)
    probs = probs * (1 + wind_effect)
    
    return probs

def compute_dispersal_matrix(config=None):
    """
    Compute the dispersal matrix once for all species.
    Returns a dictionary of dispersal matrices for each species.
    Optimized for efficient matrix operations and potential GPU usage.
    """
    if config is None:
        config = IBMConfig()
        
    # Calculate distance matrix and angles once
    distances, angles = calculate_distance_matrix(config)
    
    # Generate habitat quality once
    habitat = generate_habitat_quality(config)
    habitat = np.clip(habitat, 0.1, 10.0)
    
    # Initialize dictionary to store dispersal matrices for each species
    dispersal_matrices = {}
    
    # For each species, compute its dispersal matrix once
    for s in range(len(config.SPECIES_DISPERSAL_PARAMS['kernel_type'])):
        kernel_type = config.SPECIES_DISPERSAL_PARAMS['kernel_type'][s]
        kernel_params = config.SPECIES_DISPERSAL_PARAMS['kernel_params'][s]
        seed_size = config.SEED_SIZES[s]
        
        # Calculate dispersal probabilities with all effects
        dispersal_probs = dispersal_kernel_vectorized(
            distances, angles, kernel_type, kernel_params, seed_size, config
        )
        
        # Normalize dispersal probabilities with stability
        row_sums = np.sum(dispersal_probs, axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-10, 1e10)
        dispersal_probs = np.divide(dispersal_probs, row_sums, where=row_sums!=0)
        
        # Apply habitat quality effect
        _, establishment_factor = calculate_seed_size_effects(seed_size, config)
        habitat_effect = np.clip(habitat.reshape(-1, 1) * establishment_factor, 0.1, 10.0)
        
        # Store the normalized dispersal matrix with habitat effects
        dispersal_matrices[s] = {
            'matrix': dispersal_probs,
            'habitat_effect': habitat_effect,
            'dispersal_rate': config.DISPERSAL_RATE
        }
    
    return dispersal_matrices

def compute_dispersal(B, config=None, dispersal_matrices=None):
    """
    Compute the dispersal flux for a biomass array B using pre-computed dispersal matrices.
    Optimized to use efficient matrix operations and allow for potential GPU acceleration.
    """
    if config is None:
        config = IBMConfig()
        
    if B.ndim != 3:
        raise ValueError("Input array B must have shape (S, NUM_PATCHES_Y, NUM_PATCHES_X)")
    
    # Initialize flux arrays
    outgoing_flux = np.zeros_like(B)
    incoming_flux = np.zeros_like(B)
    
    # Reshape B for efficient computation with bounds
    B_reshaped = np.clip(B.reshape(B.shape[0], -1), 1e-10, 1e10)
    
    # For each species
    for s in range(B.shape[0]):
        # Get pre-computed dispersal matrix and effects
        if dispersal_matrices is not None and s in dispersal_matrices:
            matrix_data = dispersal_matrices[s]
            D = matrix_data['matrix']
            habitat_effect = matrix_data['habitat_effect']
            dispersal_rate = matrix_data['dispersal_rate']
        else:
            # Fallback to computing dispersal matrix for this species
            distances, angles = calculate_distance_matrix(config)
            kernel_type = config.SPECIES_DISPERSAL_PARAMS['kernel_type'][s]
            kernel_params = config.SPECIES_DISPERSAL_PARAMS['kernel_params'][s]
            seed_size = config.SEED_SIZES[s]
            
            D = dispersal_kernel_vectorized(
                distances, angles, kernel_type, kernel_params, seed_size, config
            )
            row_sums = np.sum(D, axis=1, keepdims=True)
            row_sums = np.clip(row_sums, 1e-10, 1e10)
            D = np.divide(D, row_sums, where=row_sums!=0)
            
            _, establishment_factor = calculate_seed_size_effects(seed_size, config)
            habitat = generate_habitat_quality(config)
            habitat = np.clip(habitat, 0.1, 10.0)
            habitat_effect = np.clip(habitat.reshape(-1, 1) * establishment_factor, 0.1, 10.0)
            dispersal_rate = config.DISPERSAL_RATE
        
        # Get biomass for this species
        B_s = B_reshaped[s]
        
        # Compute fluxes using optimized matrix operations
        # First, compute the effective dispersal matrix with all effects
        D_effective = dispersal_rate * D * habitat_effect
        
        # Compute outgoing flux using matrix multiplication
        outgoing = np.sum(D_effective * B_s.reshape(-1, 1), axis=1)
        
        # Compute incoming flux using matrix multiplication
        incoming = np.sum(D_effective * B_s.reshape(1, -1), axis=0)
        
        # Reshape and clip fluxes
        outgoing_flux[s] = np.clip(
            outgoing.reshape(config.NUM_PATCHES_Y, config.NUM_PATCHES_X),
            0, 1e10
        )
        incoming_flux[s] = np.clip(
            incoming.reshape(config.NUM_PATCHES_Y, config.NUM_PATCHES_X),
            0, 1e10
        )
    
    return outgoing_flux, incoming_flux
