############################################
# models_ibm.py
############################################
from accelerator import np
import logging
import sys
from typing import Optional, Literal
from config import (
    BODY_MASS, MORTALITY_RATE, STEP_SIZE, TMAX,
    RECORDING_STEP_SIZE, NUM_PATCHES_X, NUM_PATCHES_Y,
    CONNECTANCE, INTERACTION_STRENGTH
)
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX
from environment import generate_spatial_r
from config import LONG_DISTANCE_PROB

logger = logging.getLogger(__name__)

class IBMModel:
    def __init__(self,
                 r, C=None, initial_N=None,
                 r_field=None, length_scale=None, var_r=None, seed_field=None,
                 tmax=None, record_step=None,
                 record_mode: Literal['full', 'mean', 'none']='full',
                 seed=None, # Accepted for API compatibility
                 dispersal_type='propagule', dispersal_away_rate=None):
        
        self.S = len(r)
        self.N_patches = NUM_PATCHES_X * NUM_PATCHES_Y
        self.shape_3d = (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
        
        # 1. Structure Generation (Uses global RNG, does not reset seed)
        if C is None:
            # We use np.random (global state) instead of a local rng 
            # to ensure the sequence continues from previous steps.
            C = np.eye(self.S, dtype=float)
            # Create random mask and values on device directly or via numpy
            # Using loop for clarity as per your original code, but relying on global state
            for i in range(self.S):
                for j in range(self.S):
                    if i != j and np.random.rand() < CONNECTANCE:
                        C[i,j] = INTERACTION_STRENGTH * np.random.rand()
        self.C = np.asarray(C, float)

        # 2. Environment Setup
        if r_field is None:
            if (length_scale is not None) and (var_r is not None):
                self.r_field = generate_spatial_r(
                    self.S, NUM_PATCHES_Y, NUM_PATCHES_X,
                    length_scale, r, var_r, seed=seed_field
                )
            else:
                if self.S > 0:
                    # Uniform r field
                    self.r_field = np.broadcast_to(
                        np.asarray(r, float).reshape(self.S,1,1),
                        (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
                    )
                else:
                    self.r_field = np.zeros((0, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=float)
        else:
            self.r_field = r_field

        self.r_field = np.asarray(self.r_field, dtype=float).copy()
        self.r_flat = self.r_field.reshape(self.S, self.N_patches)
        
        self.tmax = tmax if tmax is not None else TMAX
        self.nsteps = int(self.tmax / STEP_SIZE)
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        self.record_mode = record_mode.lower()
        self.dispersal_type = dispersal_type
        
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate
        else:
            # Reshape to match 2D spatial grid for broadcasting later
            self.dispersal_away_rate = \
                np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten(). \
                reshape((NUM_PATCHES_Y, NUM_PATCHES_X))

        # FIX: REMOVED SEED RESET
        # np.random.seed(seed) <--- This was causing the "Stubborn Seed" issue

        # 3. State Initialization
        if initial_N is not None:
            # FIX: Force 3D shape (S, Y, X) to prevent IndexError in dispersal.py
            self.N = np.asarray(initial_N, dtype=int).reshape(self.shape_3d)
        else:
            if self.S > 0:
                init_biomass = BODY_MASS / 10
                self.N = np.full(
                    self.shape_3d,
                    max(1, int(init_biomass / BODY_MASS)),
                    dtype=int
                )
            else:
                self.N = np.zeros((0, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=int)

        self.nrecords = max(1, self.nsteps // self.record_step)
        
        if self.record_mode == "full":
            self.trajectory = np.zeros((self.nrecords, self.S,
                                      NUM_PATCHES_Y, NUM_PATCHES_X), dtype=int)
        elif self.record_mode == "mean":
            self.trajectory = np.zeros((self.nrecords, self.S), dtype=float)
        else:
            self.trajectory = None

    def run(self):
        logger.info(f"Starting IBM simulation. S={self.S}, Tmax={self.tmax}")
        if self.S == 0:
             return self.trajectory
             
        record_idx = 0
        init_pop = self.N.sum().item()
        logger.info(f"[DEBUG] Initial Population Total: {init_pop}")

        for s in range(self.nsteps):
            
            # Biomass conversion
            B = self.N.astype(float) * BODY_MASS
            
            # Dispersal (Requires B to be 3D: S, Y, X)
            incoming_flux = compute_dispersal(B)
            
            # Local Growth (Requires Flattened: S, N_patches)
            B_reshaped = B.reshape(self.S, -1)
            local_growth_flat = self.r_flat - (self.C @ B_reshaped)
            local_growth_rates = local_growth_flat.reshape(B.shape)
            
            if self.dispersal_type == 'adult':
                # dispersal_away_rate is (Y,X), broadcasts to (S,Y,X)
                local_growth_rates -= self.dispersal_away_rate
            
            # Mortality calculations
            fast_dying = local_growth_rates < (-MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -MORTALITY_RATE
            
            if self.dispersal_type == 'adult':
                full_mortality = full_mortality + self.dispersal_away_rate
            
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            
            if s == 0:
                logger.info(f"[DEBUG Step 0] Survival Prob avg: {survival_prob.mean().item():.4f}")

            # Stochastic updates
            new_N = np.random.binomial(self.N, survival_prob)
            
            birth_lambda = (np.exp((local_growth_rates + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
            
            # Update state
            self.N = (new_N + birth_values + incoming).astype(int)

            # Sanity check (optional, usually safe with these ops)
            # if (self.N < 0).any().item(): sys.exit("Negative abundances!!")
            
            # Recording
            if (s+1) % self.record_step == 0:
                if record_idx < self.nrecords:
                    if self.record_mode == "full":
                        self.trajectory[record_idx] = self.N 
                    elif self.record_mode == "mean":
                        self.trajectory[record_idx] = (self.N * BODY_MASS).mean((1,2))
                    record_idx += 1

        logger.info("IBM simulation completed.")
        return self.trajectory
