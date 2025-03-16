############################################
# models_ibm_multi.py
############################################
"""
IBM Model for spatial multi‐patch dynamics with dispersal.
Local dynamics are updated in each patch using patch‐specific intrinsic growth rates,
then quantum‐inspired dispersal is applied over the entire grid.
Additionally, a spatial diagnostic (dispersal flux) is recorded at each recording step.
"""

import numpy as np
import logging
from config import BODY_MASS, INV, MORTALITY_RATE, STEP_SIZE, TMAX, RECORDING_STEP_SIZE, NUM_PATCHES_X, NUM_PATCHES_Y
from dispersal import compute_spatial_flux

logger = logging.getLogger(__name__)

class IBMMultiPatchModel:
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123, hetero_strength=0.05):
        """
        Initialize the IBM multi‐patch model.
        
        Parameters:
            r : 1D numpy array of intrinsic growth rates (length S)
            C : 2D numpy array, competition matrix (S x S)
            nsteps : number of simulation steps (default: TMAX)
            record_step : interval for recording the state (default: RECORDING_STEP_SIZE)
            seed : random seed for reproducibility
            hetero_strength : amplitude of random perturbation for spatial heterogeneity (default: 0.05)
                              (Higher values produce greater spatial variability in growth rates.)
        """
        self.global_r = r  # store the global growth rates
        self.C = C
        self.S = len(r)
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        np.random.seed(seed)
        
        # Create a spatial field of intrinsic growth rates to introduce heterogeneity
        self.r_field = r + hetero_strength * np.random.randn(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
        
        # Initialize species counts so that each species starts with a small biomass (BODY_MASS/10)
        init_biomass = BODY_MASS / 10
        self.N = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), init_biomass, dtype=int)
        
        # Storage for main output: biomass in each patch (after dispersal)
        self.nrecords = self.nsteps // self.record_step
        self.trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        # Record the dispersal flux applied at each recording step
        self.dispersal_flux_trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
    
    def run(self):
        logger.info("Starting IBM multi-patch simulation with quantum-inspired dispersal and spatial heterogeneity...")
        record_idx = 0
        for s in range(self.nsteps):
            # Convert counts to biomass
            B_all = self.N * BODY_MASS
            
            # Compute local growth for each patch using vectorized operations
            local_growth = self.r_field - np.einsum('ijk,lk->ijl', B_all, self.C)
            
            # Apply fast mortality adjustments
            fast_dying = local_growth < (-MORTALITY_RATE)
            full_mortality = np.where(fast_dying, -local_growth, MORTALITY_RATE)
            local_growth[fast_dying] = -MORTALITY_RATE
            
            # Compute survival probabilities
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            
            # Compute birth and invasion values
            birth_lambda = (np.exp((local_growth + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_vals = np.random.poisson(birth_lambda)
            invasion_vals = np.random.poisson(INV * (STEP_SIZE / BODY_MASS), size=(NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
  
            # Update species counts
            self.N = new_N + birth_vals + invasion_vals
            
            # Dispersal update
            flux = compute_spatial_flux(B_all)
            
            # Record the dispersal flux diagnostic at recording steps
            if (s + 1) % self.record_step == 0:
                self.dispersal_flux_trajectory[record_idx, :, :, :] = flux
            
            # Update biomass by adding the dispersal flux
            B_updated = B_all + flux
            self.N = np.floor(B_updated / BODY_MASS).astype(int)
            
            # Record state after dispersal
            if (s + 1) % self.record_step == 0:
                self.trajectory[record_idx, :, :, :] = self.N * BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"IBM multi-patch progress: {record_idx}/{self.nrecords} records recorded.")
        
        logger.info("IBM multi-patch simulation completed.")
        return self.trajectory, self.dispersal_flux_trajectory
