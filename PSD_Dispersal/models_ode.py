############################################
# models_ode_multi.py
############################################
"""
ODE Model extended for spatial multi-patch dynamics with dispersal.
State is represented in log-space; the derivative includes local dynamics and a dispersal term.
Additionally, spatially resolved diagnostics (growth rate, invasion rate, dispersal term)
are computed and recorded.

>> NEW: This version introduces spatial heterogeneity by creating a per-patch intrinsic 
   growth rate field (r_field) which perturbs the global r by a small random amount.
   Additionally, quantum-inspired dispersal is optionally applied if enabled in the configuration.
"""

import numpy as np
import logging
from config import BODY_MASS, INV, TMAX, RECORDING_STEP_SIZE, RTOL, ATOL, NUM_PATCHES_X, NUM_PATCHES_Y, QUANTUM_DISPERSAL
# Import the unified dispersal function.
from dispersal import compute_spatial_flux
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

logger = logging.getLogger(__name__)

class ODEMultiPatchModel:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123, hetero_strength=0.05):
        """
        Initialize the ODE multi-patch model.
        
        Parameters:
          r : 1D array of global intrinsic growth rates (length S)
          C : 2D competition matrix (S x S)
          tmax : maximum simulation time (default: TMAX)
          record_step : recording interval (default: RECORDING_STEP_SIZE)
          seed : random seed for reproducibility
          hetero_strength : amplitude of random perturbation to introduce spatial heterogeneity (default: 0.05)
        """
        np.random.seed(seed)
        self.r = np.asarray(r, dtype=float)    # Global r, shape: (S,)
        self.C = np.asarray(C, dtype=float)      # Competition matrix, shape: (S, S)
        self.S = len(self.r)
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        init_biomass = BODY_MASS / 10
        # The state (logB) is stored in an array of shape (NUM_PATCHES_Y, NUM_PATCHES_X, S)
        self.logB = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.log(init_biomass))
        
        # >> NEW: Create a per-patch intrinsic growth rate field to introduce spatial heterogeneity.
        # Each patch receives: global r + hetero_strength * random noise.
        self.r_field = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X, self.S))
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                self.r_field[i, j, :] = self.r + hetero_strength * np.random.randn(self.S)
        # >> END NEW

        # Prepare output arrays for the main trajectory.
        self.nrecords = self.tmax // self.record_step if self.record_step > 0 else 1
        self.trajectory = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.time_points = np.zeros(self.nrecords+1)
        self.record_idx = 0
        
        # Spatially resolved diagnostics:
        self.growth_rate_traj   = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.invasion_rate_traj = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.dispersal_term_traj= np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
    
    def _deriv(self, t, logB_flat):
        """
        Compute the derivative d(logB)/dt for the entire grid.
        """
        # Reshape state back to (NUM_PATCHES_Y, NUM_PATCHES_X, S)
        logB = logB_flat.reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
        B = np.exp(logB)
        dlogB = np.zeros_like(logB)
        
        # Compute local dynamics for each patch.
        for i in range(NUM_PATCHES_Y):
            for j in range(NUM_PATCHES_X):
                # >> CHANGE: Use patch-specific intrinsic growth rate from r_field.
                local_growth = self.r_field[i, j, :] - self.C.dot(B[i, j, :])
                # >> END CHANGE
                # Invasion term: INV * exp(-logB)
                invasion = INV * np.exp(-logB[i, j, :])
                dlogB[i, j, :] = local_growth + invasion
        
        # Compute dispersal flux over the entire grid using unified dispersal.
        flux = compute_spatial_flux(B)
        # Convert flux to contribution in log-space: d(logB) = flux / B.
        dlogB += flux / (B + 1e-12)
        
        return dlogB.flatten()
    
    def run(self):
        logger.info("Starting ODE multi-patch simulation with Assimulo...")
        y0 = self.logB.flatten()
        problem = Explicit_Problem(self._deriv, y0, 0.0)
        problem.name = 'ODE Multi-patch Model'
        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        solver.linear_solver = 'SPGMR'
        solver.rtol = RTOL
        solver.atol = ATOL
        
        times = np.arange(0, self.tmax + self.record_step, self.record_step, dtype=float)
        self.trajectory[0, :, :, :] = np.exp(self.logB)  # initial biomass in each patch
        self.time_points[0] = 0.0
        self.record_idx = 1
        
        # Loop over time steps for recording.
        for t_next in times[1:]:
            solver.simulate(t_next)
            y_sol = solver.y
            self.logB = y_sol.reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
            B = np.exp(self.logB)
            self.trajectory[self.record_idx, :, :, :] = B
            self.time_points[self.record_idx] = t_next
            
            # Compute diagnostics for each patch using unified dispersal.
            flux = compute_spatial_flux(B)
            for i in range(NUM_PATCHES_Y):
                for j in range(NUM_PATCHES_X):
                    # >> CHANGE: Use patch-specific intrinsic growth rate from r_field.
                    local_growth = self.r_field[i, j, :] - self.C.dot(B[i, j, :])
                    # >> END CHANGE
                    invasion = INV * np.exp(-self.logB[i, j, :])
                    self.growth_rate_traj[self.record_idx, i, j, :] = local_growth
                    self.invasion_rate_traj[self.record_idx, i, j, :] = invasion
                    self.dispersal_term_traj[self.record_idx, i, j, :] = flux[i, j, :] / (B[i, j, :] + 1e-12)
            
            self.record_idx += 1
            logger.info(f"ODE multi-patch progress: t={t_next}, record {self.record_idx}/{self.nrecords+1}")
            if t_next >= self.tmax:
                break
        
        logger.info("ODE multi-patch simulation completed.")
        return (self.time_points[:self.record_idx],
                self.trajectory[:self.record_idx, :, :, :],
                self.growth_rate_traj[:self.record_idx, :, :, :],
                self.invasion_rate_traj[:self.record_idx, :, :, :],
                self.dispersal_term_traj[:self.record_idx, :, :, :])
