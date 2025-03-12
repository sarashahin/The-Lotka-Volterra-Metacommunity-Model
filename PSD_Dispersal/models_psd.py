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
# Use the unified dispersal function that selects classical or quantum-inspired dispersal.
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
        # Initialize state variables for each patch:
        # logB: log biomass; waiting: boolean flag; poisson_clock: for invasion events.
        self.logB = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.log(init_biomass))
        self.waiting = np.ones((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), dtype=bool)
        self.poisson_clock = np.log(np.random.rand(NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        
        self.nrecords = self.nsteps // self.record_step
        # Main output: logB trajectory.
        self.trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.wait_trajectory = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=bool)
        
        # Spatial diagnostics (per patch and species for each record):
        self.growth_rate_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.invasion_rate_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        self.dispersal_term_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        # >> NEW: Record dispersal flux at each recording step.
        self.dispersal_flux_traj = np.full((self.nrecords, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.nan, dtype=float)
        
        # >> NEW: Create a spatial field of intrinsic growth rates to introduce heterogeneity.
        # Each patch receives a slightly different growth rate vector.
        self.r_field = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                # Global r perturbed by random noise scaled by hetero_strength.
                self.r_field[i, j, :] = r + hetero_strength * np.random.randn(self.S)
        # >> END NEW

    def run(self):
        logger.info("Starting PSD multi-patch simulation with dispersal and spatial diagnostics...")
        record_idx = 0
        for s in range(self.nsteps):
            # Loop over each patch to update local dynamics.
            for i in range(NUM_PATCHES_Y):
                for j in range(NUM_PATCHES_X):
                    # Compute effective biomass (waiting species are not established yet).
                    B = np.exp(self.logB[i, j, :]) * (~self.waiting[i, j, :])
                    # >> CHANGE: Use patch-specific growth rates from r_field instead of global r.
                    local_growth = self.r_field[i, j, :] - self.C.dot(B)
                    # >> END CHANGE
                    
                    was_waiting = self.waiting[i, j, :].copy()
                    self.waiting[i, j, :] = (local_growth > 0) & (B < BODY_MASS)
                    new_waiting = np.where(self.waiting[i, j, :] & (~was_waiting))[0]
                    
                    for nw in new_waiting:
                        est_prob = 1.0 / (1.0 + MORTALITY_RATE / local_growth[nw]) if local_growth[nw] != 0 else 0
                        self.poisson_clock[i, j, nw] += B[nw] / BODY_MASS * est_prob
                        if self.poisson_clock[i, j, nw] > 0:
                            n_est = 1
                            while True:
                                self.poisson_clock[i, j, nw] += np.log(np.random.rand())
                                if self.poisson_clock[i, j, nw] <= 0:
                                    break
                                n_est += 1
                            self.waiting[i, j, nw] = False
                            self.logB[i, j, nw] = np.log(BODY_MASS * n_est / est_prob)
                    
                    stopped_waiting = np.where((~self.waiting[i, j, :]) & was_waiting)[0]
                    for sw in stopped_waiting:
                        self.logB[i, j, sw] = np.log(INV * STEP_SIZE / 2)
                    
                    dB = local_growth + np.exp(np.log(INV) - self.logB[i, j, :])
                    dB[self.waiting[i, j, :]] = 0
                    
                    waiting_list = np.where(self.waiting[i, j, :])[0]
                    if waiting_list.size > 0:
                        est_prob_list = local_growth[waiting_list] / (local_growth[waiting_list] + MORTALITY_RATE)
                        inv_est_prob = INV * (STEP_SIZE / BODY_MASS) * est_prob_list
                        self.poisson_clock[i, j, waiting_list] += inv_est_prob
                        invades = np.where(self.poisson_clock[i, j, waiting_list] > 0)[0]
                        if invades.size > 0:
                            for idx_local in invades:
                                species_idx = waiting_list[idx_local]
                                n_est = 1
                                while True:
                                    self.poisson_clock[i, j, species_idx] += np.log(np.random.rand())
                                    if self.poisson_clock[i, j, species_idx] <= 0:
                                        break
                                    n_est += 1
                                self.logB[i, j, species_idx] = np.log(BODY_MASS * n_est / est_prob_list[idx_local])
                                if self.logB[i, j, species_idx] > 0:
                                    self.logB[i, j, species_idx] = 0
                                dB[species_idx] *= 0.5
                                self.waiting[i, j, species_idx] = False
                    self.logB[i, j, :] += dB * STEP_SIZE
                    self.logB[i, j, :] = np.minimum(self.logB[i, j, :], 0)
            
            # --- Apply dispersal update over the grid ---
            B_all = np.exp(self.logB)
            # Use the unified dispersal function to automatically choose quantum-inspired or classical dispersal.
            flux = compute_spatial_flux(B_all)
            B_updated = B_all + flux
            self.logB = np.log(np.maximum(B_updated, 1e-12))
            
            # Record state and compute diagnostics at recording steps.
            if (s + 1) % self.record_step == 0:
                self.trajectory[record_idx, :, :, :] = self.logB.copy()
                self.wait_trajectory[record_idx, :, :, :] = self.waiting.copy()
                # Record the dispersal flux.
                self.dispersal_flux_traj[record_idx, :, :, :] = flux
                
                # Compute diagnostics per patch.
                B_current = np.exp(self.logB)
                for i in range(NUM_PATCHES_Y):
                    for j in range(NUM_PATCHES_X):
                        # >> CHANGE: Use patch-specific growth rates.
                        local_growth = self.r_field[i, j, :] - self.C.dot(B_current[i, j, :])
                        # >> END CHANGE
                        self.growth_rate_traj[record_idx, i, j, :] = local_growth
                        self.invasion_rate_traj[record_idx, i, j, :] = INV * np.exp(-self.logB[i, j, :])
                        self.dispersal_term_traj[record_idx, i, j, :] = flux[i, j, :] / (B_current[i, j, :] + 1e-12)
                
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"PSD multi-patch progress: {record_idx}/{self.nrecords} records recorded.")
        
        logger.info("PSD multi-patch simulation completed.")
        return (self.trajectory, self.wait_trajectory,
                self.growth_rate_traj, self.invasion_rate_traj,
                self.dispersal_term_traj, self.dispersal_flux_traj)
