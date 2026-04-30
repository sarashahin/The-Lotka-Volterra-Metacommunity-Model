# ############################################
# models_psd2.py  (ULTRA-FAST pure NumPy version)
# ############################################
"""
PSD2 approach optimized for MAXIMUM SPEED.

Uses pure NumPy Euler integration - no Assimulo overhead.
This eliminates all solver initialization and event handling costs.
"""

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "4")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
_os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import logging
import time

from config import (
    BODY_MASS, MORTALITY_RATE,
    STEP_SIZE, RECORDING_STEP_SIZE,
    TMAX, INV)

logger = logging.getLogger(__name__)


class PSD2Model:
    """
    Ultra-fast PSD model using pure NumPy vectorized Euler integration.
    No Assimulo overhead, no expensive event detection.
    """
    
    def __init__(self, r, C, tmax=None, record_step=None, seed=123):
        np.random.seed(seed)
        self.runtime_seconds = None

        # Model data
        self.r = np.asarray(r, dtype=np.float64).ravel()
        self.C = np.asarray(C, dtype=np.float64)
        self.S = self.r.size

        self.tmax = int(tmax if tmax is not None else TMAX)
        self.record_step = int(record_step if record_step is not None else RECORDING_STEP_SIZE)
        self.dt = 10.0  # Integration step size

        # Precompute
        self.C_diag = np.diag(self.C).copy()
        self._inv_over_BM = INV / BODY_MASS

        # Output arrays
        self.nrecords = self.tmax // self.record_step
        self.trajectory = np.zeros((self.nrecords + 1, self.S), dtype=np.float64)
        self.wait_trajectory = np.zeros((self.nrecords + 1, self.S), dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1, dtype=np.float64)
        self.poisson_clock_traj = np.zeros((self.nrecords + 1, self.S), dtype=np.float64)
        self.growth_rate_traj = np.zeros((self.nrecords + 1, self.S), dtype=np.float64)
        self.invasion_rate_traj = np.zeros((self.nrecords + 1, self.S), dtype=np.float64)
        self.establishment_prob_traj = np.zeros((self.nrecords + 1, self.S), dtype=np.float64)

        if logger.isEnabledFor(logging.INFO):
            logger.info("PSD2Model init: S=%d, tmax=%d, record_step=%d, dt=%.2f",
                        self.S, self.tmax, self.record_step, self.dt)

    def run(self):
        """
        Run PSD2 simulation using vectorized NumPy Euler integration.
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info("Starting PSD2 simulation (pure NumPy)...")
        t0 = time.perf_counter()

        # Local references for speed
        S = self.S
        r = self.r
        C = self.C
        C_diag = self.C_diag
        dt = self.dt
        inv_over_BM = self._inv_over_BM
        
        # State variables
        init_biomass = BODY_MASS / 2.0
        logB = np.full(S, np.log(init_biomass), dtype=np.float64)
        pclock = np.log(np.random.rand(S))
        waiting = np.ones(S, dtype=bool)  # All start in S-state
        
        # Buffers to avoid allocation
        B = np.empty(S, dtype=np.float64)
        local_growth = np.empty(S, dtype=np.float64)
        non_self_growth = np.empty(S, dtype=np.float64)
        dlogB = np.empty(S, dtype=np.float64)
        dpclock = np.empty(S, dtype=np.float64)
        est_prob = np.empty(S, dtype=np.float64)
        
        record_idx = 0
        nsteps = self.tmax
        record_step = self.record_step

        for step in range(nsteps + 1):
            # Compute B and growth rates
            np.exp(logB, out=B)
            np.copyto(local_growth, r)
            local_growth -= C.dot(B)
            np.multiply(C_diag, B, out=non_self_growth)
            non_self_growth += local_growth  # ĝ_i = r_i - sum_{j≠i} C_ij B_j
            
            # Record state
            if step % record_step == 0 and record_idx <= self.nrecords:
                # Biomass (zero for waiting species)
                self.trajectory[record_idx] = np.where(waiting, 0.0, B)
                self.wait_trajectory[record_idx] = waiting.copy()
                self.time_points[record_idx] = step
                self.growth_rate_traj[record_idx] = non_self_growth.copy()
                self.poisson_clock_traj[record_idx] = pclock.copy()
                
                # Establishment probability
                pos_growth = non_self_growth > 0
                est_prob.fill(0.0)
                est_prob[pos_growth] = non_self_growth[pos_growth] / (non_self_growth[pos_growth] + MORTALITY_RATE)
                self.establishment_prob_traj[record_idx] = est_prob.copy()
                self.invasion_rate_traj[record_idx] = inv_over_BM * est_prob
                
                record_idx += 1
                
                if record_idx % 100 == 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"PSD2: step={step}, {record_idx}/{self.nrecords+1}")

            if step >= nsteps:
                break

            # ============================================================
            # STATE TRANSITIONS (vectorized)
            # ============================================================
            
            # S -> D: Poisson clock triggers establishment
            s_to_d = waiting & (pclock >= 0) & (non_self_growth >= 0)
            if np.any(s_to_d):
                ep = non_self_growth[s_to_d] / (non_self_growth[s_to_d] + MORTALITY_RATE)
                val = BODY_MASS / np.maximum(ep, 1e-10)
                logB[s_to_d] = np.where(val > 1, 0.0, np.log(val))
                pclock[s_to_d] = 1.0
                waiting[s_to_d] = False
            
            # S -> P: Growth becomes negative while waiting
            s_to_p = waiting & (non_self_growth < 0)
            if np.any(s_to_p):
                pclock[s_to_p] = 1.0
                waiting[s_to_p] = False
            
            # P/D -> S: Growth becomes positive (probabilistic)
            # Only check non-waiting species where growth just became positive
            potential_to_s = ~waiting & (non_self_growth >= 0) & (non_self_growth < 0.1)
            if np.any(potential_to_s):
                idx_pot = np.where(potential_to_s)[0]
                for i in idx_pot:
                    c = max(0.01, abs(local_growth[i]))
                    prob_to_S = np.exp(-B[i] / (BODY_MASS * (1 + np.sqrt(np.pi / (2 * c)) * MORTALITY_RATE)))
                    if np.random.rand() < prob_to_S:
                        waiting[i] = True
                        pclock[i] = np.log(np.random.rand())
            
            # ============================================================
            # EULER UPDATE (vectorized)
            # ============================================================
            
            # dlogB/dt = local_growth + INV/B - 2*max(0, ĝ)*waiting
            np.copyto(dlogB, local_growth)
            dlogB += INV / B
            waiting_contrib = np.maximum(0, non_self_growth) * waiting
            dlogB -= 2.0 * waiting_contrib
            
            # dpclock/dt for waiting species
            dpclock.fill(0.0)
            waiting_pos = waiting & (non_self_growth > 0)
            if np.any(waiting_pos):
                ep_w = non_self_growth[waiting_pos] / (non_self_growth[waiting_pos] + MORTALITY_RATE)
                dpclock[waiting_pos] = ep_w * inv_over_BM
            
            # Update
            logB += dlogB * dt
            pclock += dpclock * dt
            
            # Clamp logB to prevent overflow
            np.clip(logB, -50, 10, out=logB)

        self.runtime_seconds = time.perf_counter() - t0
        if logger.isEnabledFor(logging.INFO):
            logger.info("PSD2 simulation completed in %.3f s.", self.runtime_seconds)

        return (self.time_points[:record_idx],
                self.trajectory[:record_idx],
                self.wait_trajectory[:record_idx],
                self.poisson_clock_traj[:record_idx],
                self.growth_rate_traj[:record_idx],
                self.invasion_rate_traj[:record_idx],
                self.establishment_prob_traj[:record_idx])
