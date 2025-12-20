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
                 r,                 # either 1D array (length S) or ignored if r_field supplied
                 C=None,
                 initial_N: Optional[np.ndarray] = None,   
                 r_field=None,      # expect spatial field of shape (S, Y, X)
                 length_scale=None, # for on‐the‐fly generation
                 var_r=None,        # for on‐the‐fly generation
                 seed_field=None,   # seed for generate_spatial_r
                 nsteps=None,
                 record_step=None,
                 record_mode: Literal['full', 'mean', 'none']='full',
                 seed=0,
                 dispersal_type='propagule',
                 dispersal_away_rate=None):
        """
        :param r: 1D array of intrinsic growth rates (length S).
        :param C: 2D competition matrix (SxS).
        :param nsteps: number of steps in simulation (default TMAX).
        :param record_step: record every record_step steps.
        :param seed: random seed for reproducibility.
        :param dispersal_type: 'adult' or 'propagule' - specifies which life stage disperses
        """
        self.S = len(r)  # number of species
        
        # --- Matrix Generation (Reverted to use local RNG for exact consistency) ---
        if C is None:
            rng = np.random.default_rng(seed)
            C = np.eye(self.S, dtype=float)
            for i in range(self.S):
                for j in range(self.S):
                    if i != j and rng.random() < CONNECTANCE:
                        C[i,j] = INTERACTION_STRENGTH * rng.random()
        self.C = np.asarray(C, float)

        # --- Environment Setup ---
        if r_field is None:
            if (length_scale is not None) and (var_r is not None):
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
            # assert r_field.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
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

        # ### CHECK: The AI removed this seed reset. I restored it to match original.
        # This forces the global RNG to the same state every time this class is initialized.
        np.random.seed(seed)

        # Initialize counts (N) for each patch
        # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
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
            # Reverted dtype to float32 to store Biomass (not Int counts)
            self.trajectory = np.full((self.nrecords, self.S,
                                        NUM_PATCHES_Y, NUM_PATCHES_X),
                                       np.nan, dtype=np.float32)
        elif self.record_mode == "mean":
            self.trajectory = np.full((self.nrecords, self.S),
                                       np.nan, dtype=np.float32)
        else:                # "none"
            self.trajectory = None
        
    def run(self):
        """
        Run the IBM simulation with multi-patch dynamics.
        """
        logger.info("Starting IBM simulation with multi-patch dynamics...")
        record_idx = 0
        
        for s in range(self.nsteps):
            
            # Convert counts to biomass for all patches
            B = self.N * BODY_MASS  # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
            
            if False & (s % 100 == 0):
                print(f"step {s}, {s/self.nsteps:.1%}, total N = {self.N.sum()}")
                
            # Compute dispersal flux between patches
            incoming_flux = np.asarray(compute_dispersal(B))  # Same shape as B
            
            # Reshape B for matrix multiplication with C
            B_reshaped = B.reshape(self.S, -1)  # Shape: (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            
            # Calculate growth rates for all patches at once
            local_growth_flat = self.r_flat - (self.C @ B_reshaped)
            local_growth_rates = local_growth_flat.reshape(B.shape)
            
            if self.dispersal_type == 'adult':
                local_growth_rates -= np.broadcast_to(self.dispersal_away_rate, local_growth_rates.shape)
            
            # Handle fast dying: localGrowthRate < - MORTALITY_RATE
            fast_dying = local_growth_rates < (-MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -MORTALITY_RATE
            
            # For adult dispersal, add the dispersal-away rate to mortality here.
            if self.dispersal_type == 'adult':
                full_mortality = full_mortality + self.dispersal_away_rate
            
            # Step
            # Death
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            
            # Birth
            birth_lambda = (np.exp((local_growth_rates + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            # Incoming dispersers are computed from the incoming_flux.
            incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
            # Update population by adding surviving individuals, new births, and incoming dispersers.
            self.N = new_N + birth_values + incoming 

            # Ensure no negative counts
            if (self.N < 0).any().item():
                sys.exit("Negative abundances!!")
            
            # ─── recording ────────────────────────────────────────────
            if (s+1) % self.record_step == 0:
                if self.record_mode == "full":
                    # Reverted to storing Biomass
                    self.trajectory[record_idx] = self.N * BODY_MASS
                elif self.record_mode == "mean":
                    self.trajectory[record_idx] = (self.N * BODY_MASS).mean((1,2))
                record_idx += 1

        logger.info("IBM simulation completed.")
        return self.trajectory
