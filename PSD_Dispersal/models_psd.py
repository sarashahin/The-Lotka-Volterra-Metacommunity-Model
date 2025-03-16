############################################
# models_psd_multi.py
############################################
"""
PSD Model extended for spatial multi-patch dynamics with dispersal.
This model uses a waiting state and discrete-time updates in log-space.
It now also computes spatially resolved diagnostics:
  - local growth rate,
  - invasion rate,
  - dispersal term.

>> NEW: To promote spatial heterogeneity, a per-patch intrinsic growth rate field is created.
     Each patch's growth rates are given by the global r plus a small random perturbation.
     Additionally, the dispersal flux is recorded at each recording step.
"""

import numpy as np
import logging
from config import BODY_MASS, INV, MORTALITY_RATE, STEP_SIZE, TMAX, RECORDING_STEP_SIZE, NUM_PATCHES_X, NUM_PATCHES_Y
from dispersal import compute_spatial_flux

logger = logging.getLogger(__name__)

class PSDMultiPatchModel:
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123, hetero_strength=0.05):
        """
        Initialize PSD multi-patch model.
        
        Parameters:
            r : 1D numpy array of global intrinsic growth rates.
            C : 2D numpy array representing the competition matrix.
            nsteps : total number of simulation steps (default: TMAX)
            record_step : record state every 'record_step' steps (default: RECORDING_STEP_SIZE)
            seed : random seed for reproducibility.
            hetero_strength : amplitude of random perturbation to create spatial heterogeneity.
                              (Default is 0.05)
        """
        self.r = r  # global base growth rate vector
        self.C = C
        self.S = len(r)
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        np.random.seed(seed)
        
        init_biomass = BODY_MASS / 10
        self.logB = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.log(init_biomass))
        self.waiting = np.ones((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), dtype=bool)
        self.poisson_clock = np.log(np.random.rand(NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        
        self.nrecords = self.nsteps // self.record_step
        self.trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.wait_trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=bool)
        
        self.growth_rate_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.invasion_rate_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.dispersal_term_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.dispersal_flux_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        
        # Vectorized creation of r_field
        self.r_field = r + hetero_strength * np.random.randn(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)

    def run(self):
        logger.info("Starting PSD multi-patch simulation with dispersal and spatial diagnostics...")
        record_idx = 0
        for s in range(self.nsteps):
            B = np.exp(self.logB) * (~self.waiting)
            local_growth = self.r_field - np.tensordot(B, self.C, axes=(2, 0))
            was_waiting = self.waiting.copy()
            self.waiting = (local_growth > 0) & (B < BODY_MASS)
            new_waiting = (self.waiting & ~was_waiting)
            
            # Vectorized Poisson clock update for new waiting species
            est_prob = np.divide(1.0, 1.0 + MORTALITY_RATE / local_growth, out=np.zeros_like(local_growth), where=local_growth!=0)
            self.poisson_clock += np.where(new_waiting, B / BODY_MASS * est_prob, 0)
            new_established = self.poisson_clock > 0
            self.poisson_clock += np.log(np.random.rand(*self.poisson_clock.shape)) * new_established
            self.waiting &= ~new_established
            self.logB = np.where(new_established, np.log(BODY_MASS * np.ceil(-self.poisson_clock) / est_prob), self.logB)
            
            # Vectorized stopping waiting species update
            stopped_waiting = ~self.waiting & was_waiting
            # Debugging print statements to inspect values
            # print("BODY_MASS:", BODY_MASS)
            # print("self.poisson_clock:", self.poisson_clock)
            # print("est_prob:", est_prob)
            
            # Ensure values positive before taking the log
            safe_values = np.maximum(BODY_MASS * np.ceil(-self.poisson_clock) / est_prob, 1e-10)
            # print("safe_values:", safe_values)

            self.logB = np.where(new_established, np.log(safe_values), self.logB)
            
            dB = local_growth + np.exp(np.log(INV) - self.logB)
            dB[self.waiting] = 0
            
            waiting_list = np.where(self.waiting)
            if waiting_list[0].size > 0:
                est_prob_list = local_growth[waiting_list] / (local_growth[waiting_list] + MORTALITY_RATE)
                inv_est_prob = INV * (STEP_SIZE / BODY_MASS) * est_prob_list
                self.poisson_clock[waiting_list] += inv_est_prob
                invades = self.poisson_clock[waiting_list] > 0
                if invades.any():
                    species_idx = waiting_list[2]
                    n_est = np.ones_like(species_idx)
                    while True:
                        self.poisson_clock[waiting_list] += np.log(np.random.rand(*species_idx.shape))
                        if not (self.poisson_clock[waiting_list] > 0).any():
                            break
                        n_est += 1
                    self.logB[waiting_list] = np.log(BODY_MASS * n_est / est_prob_list)
                    self.logB = np.where(self.logB > 0, 0, self.logB)
                    dB[waiting_list] *= 0.5
                    self.waiting[waiting_list] = False
            
            self.logB += dB * STEP_SIZE
            self.logB = np.minimum(self.logB, 0)
            
            # Apply dispersal update over the grid
            B_all = np.exp(self.logB)
            flux = compute_spatial_flux(B_all)
            B_updated = B_all + flux
            self.logB = np.log(np.maximum(B_updated, 1e-12))
            
            if (s + 1) % self.record_step == 0:
                self.trajectory[record_idx] = self.logB.copy()
                self.wait_trajectory[record_idx] = self.waiting.copy()
                self.dispersal_flux_traj[record_idx] = flux
                
                B_current = np.exp(self.logB)
                local_growth = self.r_field - np.tensordot(B_current, self.C, axes=(2, 0))
                self.growth_rate_traj[record_idx] = local_growth
                self.invasion_rate_traj[record_idx] = INV * np.exp(-self.logB)
                self.dispersal_term_traj[record_idx] = flux / (B_current + 1e-12)
                
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"PSD multi-patch progress: {record_idx}/{self.nrecords} records recorded.")
        
        logger.info("PSD multi-patch simulation completed.")
        return (self.trajectory, self.wait_trajectory,
                self.growth_rate_traj, self.invasion_rate_traj,
                self.dispersal_term_traj, self.dispersal_flux_traj)
