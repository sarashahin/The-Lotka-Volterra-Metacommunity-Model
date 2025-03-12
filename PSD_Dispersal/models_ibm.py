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
# Import the unified dispersal function.
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
        
        # --- NEW: Create a spatial field of intrinsic growth rates to introduce heterogeneity ---
        # Each patch gets its own growth rate vector: global r plus a small random noise.
        self.r_field = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                # Introduce heterogeneity by adding noise scaled by hetero_strength.
                self.r_field[i, j, :] = r + hetero_strength * np.random.randn(self.S)
        # --- END NEW ---
        
        # Initialize species counts so that each species starts with a small biomass (BODY_MASS/10).
        init_biomass = BODY_MASS / 10
        # initial_count = int(np.ceil(init_biomass / BODY_MASS))
        # State array: (NUM_PATCHES_Y, NUM_PATCHES_X, S)
        self.N = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), init_biomass, dtype=int)
        
        # Storage for main output: biomass in each patch (after dispersal).
        self.nrecords = self.nsteps // self.record_step
        self.trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        # Record the dispersal flux applied at each recording step.
        self.dispersal_flux_trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
    
    def run(self):
        logger.info("Starting IBM multi-patch simulation with quantum-inspired dispersal and spatial heterogeneity...")
        record_idx = 0
        for s in range(self.nsteps):
            # --- Local Dynamics Update (per patch) ---
            for i in range(NUM_PATCHES_Y):
                for j in range(NUM_PATCHES_X):
                    # Convert counts to biomass.
                    B_patch = self.N[i, j, :] * BODY_MASS
                    # --- CHANGE: Use the patch-specific intrinsic growth rates from r_field ---
                    local_growth = self.r_field[i, j, :] - self.C.dot(B_patch)
                    # --- END CHANGE ---
                    
                    # Apply fast mortality adjustments.
                    fast_dying = local_growth < (-MORTALITY_RATE)
                    full_mortality = np.full(self.S, MORTALITY_RATE)
                    full_mortality[fast_dying] = -local_growth[fast_dying]
                    local_growth[fast_dying] = -MORTALITY_RATE
                    
                    survival_prob = np.exp(-full_mortality * STEP_SIZE)
                    new_N = np.random.binomial(self.N[i, j, :], survival_prob)
                    
                    birth_lambda = (np.exp((local_growth + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
                    birth_vals = np.random.poisson(birth_lambda)
                    
                    invasion_vals = np.random.poisson(INV * (STEP_SIZE / BODY_MASS) * np.ones(self.S))
                    
                    self.N[i, j, :] = new_N + birth_vals + invasion_vals
            
            # --- Dispersal Update ---
            # Convert the entire state to biomass.
            B_all = self.N * BODY_MASS
            
            # Use the unified function to compute the dispersal flux.
            flux = compute_spatial_flux(B_all)
            
            # Record the dispersal flux diagnostic at recording steps.
            if (s + 1) % self.record_step == 0:
                self.dispersal_flux_trajectory[record_idx, :, :, :] = flux
            # Update biomass by adding the dispersal flux.
            B_updated = B_all + flux
            # Convert updated biomass back to counts.
            self.N = np.floor(B_updated / BODY_MASS).astype(int)
            
            # Record state after dispersal.
            if (s + 1) % self.record_step == 0:
                self.trajectory[record_idx, :, :, :] = self.N * BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"IBM multi-patch progress: {record_idx}/{self.nrecords} records recorded.")
        
        logger.info("IBM multi-patch simulation completed.")
        return self.trajectory, self.dispersal_flux_trajectory
