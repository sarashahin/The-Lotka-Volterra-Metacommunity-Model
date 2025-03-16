
############################################
# models_psd2_multi.py
############################################

"""
PSD2 Model extended for spatial multi-patch dynamics with dispersal.
Uses Assimulo’s CVode solver.
The state consists of [logB, poisson_clock] flattened over all patches.
Spatially resolved diagnostics are recorded: growth rate, invasion rate, 
establishment probability, the Poisson clock, and (new) dispersal flux.

>> NEW: To introduce spatial heterogeneity, a per-patch intrinsic growth rate field 
   (r_field) is generated from the global r plus a small random perturbation.
   Additionally, quantum-inspired dispersal is optionally applied if enabled in the configuration.
"""

import numpy as np
import logging
from config import (BODY_MASS, INV, MORTALITY_RATE, TMAX, RECORDING_STEP_SIZE,
                    RTOL, ATOL, MAX_STEPS, NUM_PATCHES_X, NUM_PATCHES_Y, QUANTUM_DISPERSAL)
from dispersal import compute_spatial_flux
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

logger = logging.getLogger(__name__)

# Variant flags (adjust as needed)
variant_with_background_fluxes = True
variant_with_P_to_D_transitions = True

class PSD2MultiPatchModel:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123, hetero_strength=0.05):
        """
        Initialize the PSD2 multi-patch model.
        
        Parameters:
          r : 1D array of global intrinsic growth rates (length S)
          C : 2D competition matrix (S x S)
          tmax : maximum simulation time (default: TMAX)
          record_step : recording interval (default: RECORDING_STEP_SIZE)
          seed : random seed for reproducibility
          hetero_strength : amplitude of random perturbation for spatial heterogeneity (default: 0.05)
        """
        np.random.seed(seed)
        self.r = np.asarray(r, dtype=float).flatten()  # Global base r, shape: (S,)
        self.C = np.asarray(C, dtype=float)             # Competition matrix, shape: (S, S)
        self.S = len(self.r)
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        
        # Here we keep the same initialization as before.
        init_biomass = BODY_MASS / 10
        # Initialize state arrays for each patch with shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)
        self.logB = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.log(init_biomass))
        # >>> We keep waiting off (False) so that the derivative uses the background flux branch.
        self.waiting = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), dtype=bool)
        # <<< End waiting: all species are active.
        self.poisson_clock = np.log(np.random.rand(NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        
        self.nrecords = max(1, self.tmax // self.record_step)
        self.trajectory = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        self.wait_trajectory = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), dtype=bool)
        self.time_points = np.zeros(self.nrecords+1)
        self.record_idx = 0
        
        # Spatial diagnostics:
        self.poisson_clock_traj = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        self.growth_rate_traj   = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        self.invasion_rate_traj = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        self.establishment_prob_traj = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        # >> NEW: Array to record dispersal flux at each recording step.
        self.dispersal_flux_traj = np.zeros((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        
        # >> NEW: Create a spatial field of intrinsic growth rates to introduce heterogeneity.
        # Each patch gets its own growth rate vector by perturbing the global r.
        self.r_field = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                self.r_field[i, j, :] = self.r + hetero_strength * np.random.randn(self.S)
        # >> END NEW
        
        logger.info(f"PSD2MultiPatchModel init: S={self.S}, tmax={self.tmax}, record_step={self.record_step}")
    
    def _derivatives(self, t, y, sw):
        """
        Compute the derivatives for the state vector.
        The state is concatenated as [logB, poisson_clock] flattened over all patches.
        The derivative for logB includes both local dynamics (with waiting correction)
        and a dispersal term.
        """
        total = NUM_PATCHES_Y * NUM_PATCHES_X * self.S
        logB = y[:total].reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
        pclock = y[total:2*total].reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
        
        B = np.exp(logB)
        dlogB = np.zeros_like(logB)
        dpclock = np.zeros_like(pclock)
        
        # Compute local dynamics in each patch.
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                # Use patch-specific growth rates from r_field.
                local_growth = self.r_field[i, j, :] - self.C.dot(B[i, j, :])
                if variant_with_background_fluxes:
                    diagB = np.diag(self.C) * B[i, j, :]
                    # Since waiting is False, waiting_corr is 0.
                    waiting_corr = self.waiting[i, j, :].astype(float)
                    dlogB[i, j, :] = local_growth + INV / B[i, j, :] - 2 * waiting_corr * (local_growth + diagB)
                    # Compute dpclock as in the original formulation
                    non_self_growth = local_growth + diagB
                    non_self_growth[non_self_growth < 0] = 0
                    denom = non_self_growth + MORTALITY_RATE
                    dpclock[i, j, :] = non_self_growth / denom * (INV / BODY_MASS)
                else:
                    # Without background flux corrections:
                    dlogB[i, j, :] = local_growth + INV / B[i, j, :]
        
        # Add dispersal term: compute net flux and convert to contribution in log-space.
        flux = compute_spatial_flux(B)
        dlogB += flux / (B + 1e-12)
        
        return np.concatenate([dlogB.flatten(), dpclock.flatten()])
    
    def run(self):
        logger.info("Starting PSD2 multi-patch simulation with Assimulo...")
        total = NUM_PATCHES_Y * NUM_PATCHES_X * self.S
        y0 = np.concatenate([self.logB.flatten(), self.poisson_clock.flatten()])
        
        # Set up the Assimulo problem
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=self.waiting.flatten())
        problem.name = 'PSD2 Multi-patch Problem'
        
        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        solver.linear_solver = 'SPGMR'
        solver.rtol = RTOL
        solver.atol = ATOL
        solver.options['hmin'] = 1e-2
        solver.options['maxh'] = 10000
        solver.options['root_tol'] = 1e0
        solver.options["mxhnil"] = 5
        solver.options['maxsteps'] = MAX_STEPS
        
        record_times = np.arange(0, self.tmax + self.record_step, self.record_step)
        record_times = np.unique(record_times)
        t, y = solver(self.tmax, record_times.shape[0]-1)
        
        recIdx = np.where(np.remainder(t, self.record_step) == 0)[0]
        if recIdx.shape[0] != record_times.shape[0]:
            raise RuntimeError("Recording time steps do not match.")
        
        # For each recording time, extract the state and compute diagnostics
        for idx, rec in enumerate(recIdx):
            state = y[rec, :]
            logB_rec = state[:total].reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
            waiting_rec = state[total:2*total].reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S) < 0
            self.trajectory[idx, :, :, :] = logB_rec
            self.wait_trajectory[idx, :, :, :] = waiting_rec
            self.time_points[idx] = t[rec]
            
            # Compute diagnostics using vectorized operations
            B_rec = np.exp(logB_rec)
            flux = compute_spatial_flux(B_rec)
            self.dispersal_flux_traj[idx, :, :, :] = flux
            
            local_growth = self.r_field - np.einsum('ijk,lk->ijl', B_rec, self.C)
            self.growth_rate_traj[idx, :, :, :] = local_growth
            self.invasion_rate_traj[idx, :, :, :] = INV * np.exp(-logB_rec)
            est_prob = np.where(local_growth > 0, local_growth / (local_growth + MORTALITY_RATE), 0)
            self.establishment_prob_traj[idx, :, :, :] = est_prob
            self.poisson_clock_traj[idx, :, :, :] = state[total:2*total].reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
            
            logger.info(f"PSD2 multi-patch recorded at t={t[rec]}")
        
        logger.info("PSD2 multi-patch simulation completed.")
        return (self.time_points,
                self.trajectory,
                self.wait_trajectory,
                self.poisson_clock_traj,
                self.growth_rate_traj,
                self.invasion_rate_traj,
                self.establishment_prob_traj,
                self.dispersal_flux_traj)








# # import numpy as np
# import matplotlib.pyplot as plt


# # Define the number of species (S) and initialize parameters
# S = 10  # example: 10 species
# # Generate a random intrinsic growth rate vector (r) and competition matrix (C)
# r = np.random.uniform(0.1, 1.0, S)
# C = np.random.uniform(0, 0.5, (S, S))

# # Instantiate the PSD2 multi-patch model
# # tmax and record_step are set to smaller values for testing purposes
# psd2_model = PSD2MultiPatchModel(r, C, tmax=5000, record_step=100)

# # Run the simulation
# # The model returns: (time_points, trajectory, waiting, poisson_clock, growth_rate, invasion_rate, establishment_prob, dispersal_flux)
# time_points, trajectory, _, _, _, _, _, _ = psd2_model.run()

# # Compute biomass from logB trajectory and average over all patches and species
# # trajectory has shape (nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, S)
# biomass = np.exp(trajectory)
# avg_biomass = np.mean(biomass, axis=(1, 2, 3))  # average over patch dimensions and species

# # Plot the average biomass trajectory over time
# plt.figure(figsize=(8, 6))
# plt.plot(time_points, avg_biomass, marker='o', linestyle='-')
# plt.xlabel("Time")
# plt.ylabel("Average Biomass per Species")
# plt.title("PSD2 Trajectory: Average Biomass vs. Time")
# plt.grid(True)
# plt.show()


