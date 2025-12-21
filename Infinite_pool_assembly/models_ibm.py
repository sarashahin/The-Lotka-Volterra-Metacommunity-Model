############################################
# models_ibm.py
############################################
"""
Individual-Based Model (IBM) approach for population dynamics.
using binomial & Poisson draws each step.
Now includes multi-patch dynamics with dispersal.
"""
from accelerator import np
import logging
import sys
from typing import Optional, Literal
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    STEP_SIZE,
    TMAX,
    N_RECORDS,
    RECORDING_STEP_SIZE,
    NUM_PATCHES_X,
    NUM_PATCHES_Y,
    DISPERSAL_RATE,
    CONNECTANCE,
    INTERACTION_STRENGTH
)
from dispersal import compute_dispersal
from dispersal import LOCAL_DISPERSAL_MATRIX
from environment import generate_spatial_r
from config import LONG_DISTANCE_PROB

logger = logging.getLogger(__name__)

class IBMModel:
    """
    IBM Model with multi-patch dynamics:
    - N: integer counts for each species in each patch
    - Convert to biomass by N * BODY_MASS
    - Growth rates: r - C@B
    - Includes dispersal between patches
    - Supports both adult and propagule dispersal
    """
    def __init__(self,
                 r,                 
                 C=None,
                 initial_N: Optional[np.ndarray] = None,   
                 r_field=None,      
                 length_scale=None, 
                 var_r=None,        
                 seed_field=None,   
                 nsteps=None,
                 record_step=None,
                 record_mode: Literal['full', 'mean', 'none']='full',
                 seed=None, # <--- CHANGED: Default None, No reset to 123
                 dispersal_type='propagule',
                 dispersal_away_rate=None):
        
        # <--- REMOVED: np.random.seed(seed) to allow continuous global stream

        self.S = len(r) 
        
        # --- Matrix Generation ---
        if C is None:
            # Use local RNG if seed provided, else global
            rng = np.random.default_rng(seed) if seed is not None else np.random
            C = np.eye(self.S, dtype=float)
            for i in range(self.S):
                for j in range(self.S):
                    if i != j and rng.random() < CONNECTANCE:
                        C[i,j] = INTERACTION_STRENGTH * rng.random()
        self.C = np.asarray(C, float)

        # --- Environment Setup ---
        if r_field is None:
            if (length_scale is not None) and (var_r is not None):
                # generate_spatial_r handles seeding via seed_field
                self.r_field = generate_spatial_r(
                    self.S, NUM_PATCHES_Y, NUM_PATCHES_X,
                    length_scale, r, var_r, seed=seed_field
                )
            else:
                self.r_field = np.broadcast_to(
                    np.asarray(r, float).reshape(self.S,1,1),
                    (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
                )
        else:
            self.r_field = r_field

        self.r_field = np.asarray(self.r_field, dtype=float).copy()
        self.r_flat = self.r_field.reshape(self.S, -1)
        
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        self.record_mode = record_mode.lower()
        self.dispersal_type = dispersal_type
        
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate
        else:
            self.dispersal_away_rate = \
                np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten(). \
                reshape((NUM_PATCHES_Y, NUM_PATCHES_X))

        # Initialize counts (N) for each patch
        if initial_N is not None:
            self.N = np.asarray(initial_N).copy()
        else:
            init_biomass = BODY_MASS / 10
            self.N = np.full(
                (self.S, NUM_PATCHES_Y, NUM_PATCHES_X),
                max(1, int(init_biomass / BODY_MASS)),
                dtype=int
            )

        # --- Storage for trajectory ---
        self.nrecords = self.nsteps // self.record_step
        if self.record_mode == "full":
            self.trajectory = np.full((self.nrecords, self.S,
                                        NUM_PATCHES_Y, NUM_PATCHES_X),
                                       np.nan, dtype=np.float32)
        elif self.record_mode == "mean":
            self.trajectory = np.full((self.nrecords, self.S),
                                       np.nan, dtype=np.float32)
        else:                
            self.trajectory = None
        
    def run(self):
        logger.info("Starting IBM simulation with multi-patch dynamics...")
        record_idx = 0
        
        for s in range(self.nsteps):
            B = self.N * BODY_MASS 
            
            incoming_flux = np.asarray(compute_dispersal(B))
            
            B_reshaped = B.reshape(self.S, -1) 
            local_growth_flat = self.r_flat - (self.C @ B_reshaped)
            local_growth_rates = local_growth_flat.reshape(B.shape)
            
            if self.dispersal_type == 'adult':
                local_growth_rates -= np.broadcast_to(self.dispersal_away_rate, local_growth_rates.shape)
            
            fast_dying = local_growth_rates < (-MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -MORTALITY_RATE
            
            if self.dispersal_type == 'adult':
                full_mortality = full_mortality + self.dispersal_away_rate
            
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            
            birth_lambda = (np.exp((local_growth_rates + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
            self.N = new_N + birth_values + incoming 

            if (self.N < 0).any().item():
                sys.exit("Negative abundances!!")
            
            if (s+1) % self.record_step == 0:
                if self.record_mode == "full":
                    self.trajectory[record_idx] = self.N * BODY_MASS
                elif self.record_mode == "mean":
                    self.trajectory[record_idx] = (self.N * BODY_MASS).mean((1,2))
                record_idx += 1

        logger.info("IBM simulation completed.")
        return self.trajectory
