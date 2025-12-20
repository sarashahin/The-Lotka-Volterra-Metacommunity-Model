############################################
# models_psd2.py
############################################

import sys
import logging
from accelerator import np # PyTorch Shim
import numpy as _cpu_numpy # Standard Numpy for CPU-side control flow
from typing import Optional

from euler_simple_safe import Explicit_Problem, EulerSimpleSafe
from environment import generate_spatial_r
import math

from config import (
    BODY_MASS, MORTALITY_RATE, TMAX, RECORDING_STEP_SIZE,
    NUM_PATCHES_X, NUM_PATCHES_Y, DISPERSAL_RATE, LONG_DISTANCE_PROB,
    CONNECTANCE, INTERACTION_STRENGTH, LOG_B_CAP, RTOL, ATOL, STEP_SIZE
)

from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX

logger = logging.getLogger(__name__)

# FLOAT32 SAFETY CONSTANT
SAFE_MIN_B = 1e-30 

class PSD2Model:
    def __init__(self, r, C=None, *,
                 initial_B=None, initial_wait=None, initial_clock=None,
                 n_new: int = 0, r_field=None, length_scale=None, var_r=None,
                 seed_field=None, tmax=None, record_step=None, seed=None, # <--- Restored argument
                 dispersal_type='propagule', dispersal_away_rate=None):

        # FIX: We accept 'seed' to prevent TypeError from legacy callers,
        # but we DO NOT call np.random.seed(seed) here.
        # This ensures the global random state evolves naturally across rounds.
        # np.random.seed(seed) 

        self.S = len(r)
        self.N_patches = NUM_PATCHES_X * NUM_PATCHES_Y
        self.shape_2d = (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
        self.shape_flat = (self.S, self.N_patches)
        
        if C is None:
            # Use global rng for structure generation
            C = np.eye(self.S, dtype=float)
            mask = np.random.rand(self.S, self.S) < CONNECTANCE
            C[mask] = INTERACTION_STRENGTH * np.random.rand(np.count_nonzero(mask))
            np.fill_diagonal(C, 1.0) 

        self.C = np.asarray(C, dtype=float)
        self.C_diag = self.C.diagonal().astype(float) 

        if r_field is None:
            if length_scale is not None and var_r is not None:
                rf = generate_spatial_r(self.S, NUM_PATCHES_Y, NUM_PATCHES_X, length_scale, r, var_r, seed=seed_field)
                self.r_flat = rf.reshape(self.S, -1)
            else:
                self.r_flat = np.broadcast_to(np.asarray(r, float).reshape(self.S, 1), (self.S, self.N_patches))
        else:
            self.r_flat = r_field.reshape(self.S, -1)
            
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        # --- 3. Initial State ---
        if initial_B is not None:
            flat_B = initial_B.reshape(self.S, -1)
            flat_B = np.asarray(flat_B, dtype=float)
            self.logB = np.log(np.maximum(flat_B, SAFE_MIN_B, dtype=np.float64))
            self.logB = np.minimum(self.logB, LOG_B_CAP)
        else:
            init_biomass = BODY_MASS / 10.0
            self.logB = np.full(self.shape_flat, np.log(init_biomass))
            
        if initial_wait is not None:
            self.waiting = np.asarray(initial_wait.reshape(self.S, -1), dtype=bool)
        else:
            self.waiting = np.zeros(self.shape_flat, dtype=bool)

        if initial_clock is not None:
            self.poisson_clock = np.asarray(initial_clock.reshape(self.S, -1), dtype=float)
        else:
            self.poisson_clock = np.log(np.random.rand(*self.shape_flat))

        self.dispersal_type = dispersal_type
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate.flatten()
        else:
            self.dispersal_away_rate = np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten()

        current_B = np.exp(self.logB)
        
        local_growth, non_self_growth = self._compute_local_growth(current_B)
        
        inconsistent_waiting = self.waiting & (non_self_growth < 0)
        if np.any(inconsistent_waiting):
            self.waiting[inconsistent_waiting] = False
            self.poisson_clock[inconsistent_waiting] = 1.0

        if n_new > 0:
            self._attempt_initial_establishment(n_new, current_B, local_growth)

        self.nrecords = int(max(1, self.tmax // self.record_step))
        out_shape = (self.nrecords + 1, *self.shape_2d)
        
        self.trajectory = np.zeros(out_shape, dtype=np.float32)
        self.wait_trajectory = np.zeros(out_shape, dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)

        self.poisson_clock_traj = np.zeros(out_shape, dtype=np.float32)
        self.growth_rate_traj = np.zeros(out_shape, dtype=np.float32)
        self.invasion_rate_traj = np.zeros(out_shape, dtype=np.float32)
        self.establishment_prob_traj = np.zeros(out_shape, dtype=np.float32)

        logger.info(f"PSD2Model initialized: S={self.S}, Patches={self.N_patches}")

    def _compute_local_growth(self, B):
        competitive_loss = self.C @ B 
        local_growth = self.r_flat - competitive_loss
        if self.dispersal_type == 'adult':
            local_growth -= self.dispersal_away_rate[None, :]
        diag_term = self.C_diag[:, None] * B
        non_self_growth = local_growth + diag_term
        return local_growth, non_self_growth

    def _compute_dispersal_input(self, B):
        # FIX: Do not move to CPU. Keep B on device.
        # B is (S, N). Reshape to (S, Y, X) for dispersal.py logic
        B_3d = B.reshape(self.shape_2d)
        
        # Compute on GPU (since B is GPU tensor and LOCAL_DISPERSAL_MATRIX is GPU tensor)
        incoming_3d = compute_dispersal(B_3d)
        
        return incoming_3d.reshape(self.S, -1)

    def _get_est_prob(self, growth_rate):
        denom = growth_rate + MORTALITY_RATE
        if self.dispersal_type == 'adult':
            denom += self.dispersal_away_rate[None, :]
        prob = np.where(growth_rate > 0, growth_rate / denom, 0.0)
        return prob

    def _attempt_initial_establishment(self, n_new, current_B, local_growth):
        for j in range(self.S - n_new, self.S):
            g = local_growth[j]
            pos_mask = g > 0
            
            est = np.zeros_like(g)
            denom = g[pos_mask] + MORTALITY_RATE
            if self.dispersal_type == 'adult':
                denom += self.dispersal_away_rate[pos_mask]
            
            est[pos_mask] = g[pos_mask] / denom
            est = 1.0 - (1.0 - est)**(current_B[j] / BODY_MASS)
            
            rnd = np.random.rand(*est.shape)
            est_failures = (rnd > est) & pos_mask
            est_successes = (~est_failures) & pos_mask
            
            self.waiting[j][est_failures] = True
            n_fail = np.count_nonzero(est_failures)
            self.poisson_clock[j][est_failures] = np.log(np.random.rand(n_fail))
            
            self.logB[j][est_successes] = np.clip(
                self.logB[j][est_successes] - np.log(np.maximum(est[est_successes], 1e-10)),
                None, LOG_B_CAP
            )

    def _derivatives(self, t, y, sw):
        total = self.S * self.N_patches
        logB = y[:total].reshape(self.S, -1)
        pclock = y[total:].reshape(self.S, -1)
        
        # Float32 Safety Clamp
        logB_clamped = np.minimum(logB, LOG_B_CAP)
        B = np.exp(logB_clamped)
        safeB = np.maximum(B, SAFE_MIN_B)

        invasion_flux = self._compute_dispersal_input(B)
        local_growth, non_self_growth = self._compute_local_growth(B)

        invasion_pressure = invasion_flux / safeB
        invasion_pressure = np.clip(invasion_pressure, 0.0, 10.0)

        dlogB = local_growth + invasion_pressure

        sw_mask = np.array(sw, dtype=bool).reshape(self.S, -1)
        
        dlogB[sw_mask] = -np.maximum(0, non_self_growth[sw_mask]) + \
            invasion_pressure[sw_mask]
        
        dlogB = np.clip(dlogB, None, 10.0)

        est_prob = self._get_est_prob(non_self_growth)
        dpclock = est_prob * (invasion_flux / BODY_MASS)
        
        return np.concatenate([dlogB.flatten(), dpclock.flatten()])

    def _event_fn(self, t, y, sw):
        total = self.S * self.N_patches
        logB = y[:total].reshape(self.S, -1)
        pclock = y[total:].reshape(self.S, -1)
        
        logB_clamped = np.minimum(logB, LOG_B_CAP)
        B = np.exp(logB_clamped)

        local_growth, non_self_growth = self._compute_local_growth(B)
        return np.concatenate([non_self_growth.flatten(), pclock.flatten()])

    def _handle_event_fn(self, solver, event_info):
        total = self.S * self.N_patches
        y_vec = solver.y
        logB = y_vec[:total].reshape(self.S, -1)
        pclock = y_vec[total:].reshape(self.S, -1)
        sw = np.array(solver.sw, dtype=bool).reshape(self.S, -1)
        
        logB_clamped = np.minimum(logB, LOG_B_CAP)
        B = np.exp(logB_clamped)
        
        rootsfound = event_info[0]
        trigger_mask = (rootsfound != 0) 
        growth_trigger = trigger_mask[:total].reshape(self.S, -1)
        clock_trigger = trigger_mask[total:].reshape(self.S, -1)
        
        local_growth, non_self_growth = self._compute_local_growth(B)
        
        mask_S_to_P = growth_trigger & (non_self_growth < 0) & sw
        if np.any(mask_S_to_P):
            sw[mask_S_to_P] = False
            pclock[mask_S_to_P] = 1.0 

        mask_sweep = growth_trigger & (non_self_growth > 0) & (~sw)
        
        if np.any(mask_sweep):
            yd = self._derivatives(solver.t, solver.y, solver.sw)
            dlogB = yd[:total].reshape(self.S, -1)
            dlogB[mask_sweep] = 0.0
            
            prod = B * dlogB
            c_vals = -(self.C @ prod)
            
            eps = SAFE_MIN_B
            denom = BODY_MASS * (1.0 + np.sqrt(np.pi / (2.0 * np.maximum(c_vals, eps))) * MORTALITY_RATE)
            
            prob_remain_S = np.zeros_like(c_vals)
            valid_sweep = (c_vals > 0) & mask_sweep
            
            if np.any(valid_sweep):
                prob_remain_S[valid_sweep] = np.exp(-B[valid_sweep] / denom[valid_sweep])
                
            rnd = np.random.rand(*B.shape)
            failed_sweep = valid_sweep & (rnd <= prob_remain_S)
            successful_sweep = valid_sweep & (rnd > prob_remain_S)
            
            if np.any(successful_sweep):
                offset = -np.log1p(-prob_remain_S[successful_sweep])
                logB[successful_sweep] = np.minimum(0.0, logB[successful_sweep] + offset)
            
            if np.any(failed_sweep):
                 sw[failed_sweep] = True 
                 n_fail = np.count_nonzero(failed_sweep)
                 pclock[failed_sweep] = np.log(np.random.rand(n_fail))

        mask_clock = clock_trigger & sw
        
        if np.any(mask_clock):
            est_prob = self._get_est_prob(non_self_growth)
            valid_est = mask_clock & (est_prob > 0)
            
            if np.any(valid_est):
                val = np.divide(BODY_MASS, est_prob[valid_est])
                val = np.minimum(val, BODY_MASS) 
                logB[valid_est] = np.log(val)
                pclock[valid_est] = 1.0 
                sw[valid_est] = False 
            
            invalid_est = mask_clock & (est_prob <= 0)
            if np.any(invalid_est):
                pclock[invalid_est] = 1.0 
                sw[invalid_est] = False   

        solver.y[:total] = logB.ravel()
        solver.y[total:] = pclock.ravel()
        solver.sw = sw.ravel().tolist()
        return solver.y

    def run(self):
        logger.info("Starting PSD2 simulation (Plug-in)...")
        
        y0 = np.concatenate([self.logB.flatten(), self.poisson_clock.flatten()])
        sw0 = self.waiting.flatten().tolist()
        
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=sw0)
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        
        solver = EulerSimpleSafe(problem)
        solver.options['inith'] = STEP_SIZE

        # 1. Device tensor for storing results
        t_eval = np.arange(0, self.tmax + self.record_step, self.record_step)
        self.time_points = t_eval
        
        # 2. Host array for controlling the loop (CRITICAL FIX)
        t_eval_host = _cpu_numpy.arange(0, self.tmax + self.record_step, self.record_step)
        
        # Simulate using CPU time points
        t_out, y_out = solver.simulate(self.tmax, ncp_list=t_eval_host)
        
        total = self.S * self.N_patches
        
        # Post-processing loop using CPU control flow
        for i, t in enumerate(t_eval_host):
            idx = _cpu_numpy.abs(t_out - t).argmin()
            y_curr = y_out[idx]
            
            logB = y_curr[:total].reshape(self.S, -1)
            pclock = y_curr[total:].reshape(self.S, -1)
            
            logB_clamped = np.minimum(logB, LOG_B_CAP)
            B = np.exp(logB_clamped)
            local_g, non_self_g = self._compute_local_growth(B)
            
            # Recalculate full dispersal for recording
            inv_flux = self._compute_dispersal_input(B)
            est_prob = self._get_est_prob(non_self_g)
            
            inv_rate = inv_flux * est_prob / BODY_MASS
            
            self.trajectory[i] = B.reshape(self.shape_2d)
            self.poisson_clock_traj[i] = pclock.reshape(self.shape_2d)
            self.growth_rate_traj[i] = non_self_g.reshape(self.shape_2d)
            self.invasion_rate_traj[i] = inv_rate.reshape(self.shape_2d)
            self.establishment_prob_traj[i] = est_prob.reshape(self.shape_2d)
            
            self.wait_trajectory[i] = (pclock < 0).reshape(self.shape_2d)

        logger.info("PSD2 simulation completed.")
        
        return (
            self.time_points,
            self.trajectory,
            self.wait_trajectory,
            self.poisson_clock_traj,
            self.growth_rate_traj,
            self.invasion_rate_traj,
            self.establishment_prob_traj
        )
