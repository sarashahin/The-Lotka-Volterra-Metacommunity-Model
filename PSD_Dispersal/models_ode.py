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
from dispersal import compute_spatial_flux
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

logger = logging.getLogger(__name__)

class ODEMultiPatchModel:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123, hetero_strength=0.05):
        np.random.seed(seed)
        self.r = np.asarray(r, dtype=float)    # Global r, shape: (S,)
        self.C = np.asarray(C, dtype=float)      # Competition matrix, shape: (S, S)
        self.S = len(self.r)
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        init_biomass = BODY_MASS / 10
        self.logB = np.full((NUM_PATCHES_Y, NUM_PATCHES_X, self.S), np.log(init_biomass))
        
        self.r_field = self.r + hetero_strength * np.random.randn(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)

        self.nrecords = self.tmax // self.record_step if self.record_step > 0 else 1
        self.trajectory = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.time_points = np.zeros(self.nrecords+1)
        self.record_idx = 0
        
        self.growth_rate_traj = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.invasion_rate_traj = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
        self.dispersal_term_traj = np.full((self.nrecords+1, NUM_PATCHES_Y, NUM_PATCHES_X, self.S), 0.0)
    
    def _deriv(self, t, logB_flat):
        logB = logB_flat.reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
        B = np.exp(logB)
        dlogB = np.zeros_like(logB)
        
        local_growth = self.r_field - np.tensordot(B, self.C, axes=(2, 0))
        invasion = INV * np.exp(-logB)
        dlogB = local_growth + invasion
        
        flux = compute_spatial_flux(B)
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
        
        for t_next in times[1:]:
            solver.simulate(t_next)
            y_sol = solver.y
            self.logB = y_sol.reshape(NUM_PATCHES_Y, NUM_PATCHES_X, self.S)
            B = np.exp(self.logB)
            self.trajectory[self.record_idx, :, :, :] = B
            self.time_points[self.record_idx] = t_next
            
            flux = compute_spatial_flux(B)
            local_growth = self.r_field - np.tensordot(B, self.C, axes=(2, 0))
            invasion = INV * np.exp(-self.logB)
            self.growth_rate_traj[self.record_idx, :, :, :] = local_growth
            self.invasion_rate_traj[self.record_idx, :, :, :] = invasion
            self.dispersal_term_traj[self.record_idx, :, :, :] = flux / (B + 1e-12)
            
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
