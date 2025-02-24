
############################################
# models_psd2.py
############################################

"""
PSD2 approach with 'chunked' assimilation, iterative solver to reduce memory usage,
and more frequent logging to avoid silent long steps that risk OS kills.
"""

import numpy as np
import logging
from config import (
    BODY_MASS,
    INV,
    MORTALITY_RATE,
    TMAX,
    RECORDING_STEP_SIZE,
    RTOL,
    ATOL,
    MAX_STEPS
)
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123):
        np.random.seed(seed)

        # Force r, C to correct shapes
        self.r = np.asarray(r, dtype=float).flatten()
        self.C = np.asarray(C, dtype=float)
        self.S = len(self.r)

        self.tmax = tmax if tmax is not None else TMAX
        # We won't use record_step directly as a single chunk. We'll break tmax into smaller chunks.
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        # Initial states
        init_biomass = BODY_MASS / 10
        self.logB = np.full(self.S, np.log(init_biomass))
        self.waiting = np.ones(self.S, dtype=bool)
        self.poisson_clock = np.log(np.random.rand(self.S))

        # For storing results
        self.nrecords = max(1, self.tmax // self.record_step)
        self.trajectory = np.zeros((self.nrecords + 1, self.S))
        self.wait_trajectory = np.zeros((self.nrecords + 1, self.S), dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)
        self.record_idx = 0

        # ---- New arrays for diagnostic outputs ----
        self.poisson_clock_traj = np.zeros((self.nrecords + 1, self.S))
        self.growth_rate_traj   = np.zeros((self.nrecords + 1, self.S))
        self.invasion_rate_traj = np.zeros((self.nrecords + 1, self.S))
        self.establishment_prob_traj = np.zeros((self.nrecords + 1, self.S))
        # ---------------------------------------------

        # We'll track event sign changes ourselves
        self.last_sw = None

        # Log some initial info
        logger.info(f"PSD2Model init: S={self.S}, tmax={self.tmax}, record_step={self.record_step}")
        logger.debug(f"Initial logB: {self.logB}")
        logger.debug(f"Initial waiting: {self.waiting}")
        logger.debug(f"Initial PoissonClock: {self.poisson_clock}")
        logger.debug(f"Growth rates r: {self.r}")
        logger.debug(f"Competition Matrix C shape: {self.C.shape}")

    def _ensure_flat_y(self, y):
        """Ensure we have a 1D array of length 2*S, discarding any extra element (time)."""
        if isinstance(y, (list, tuple)) and len(y) == 2:
            part1 = np.ravel(np.asarray(y[0], dtype=float))
            part2 = np.ravel(np.asarray(y[1], dtype=float))
            flat = np.concatenate([part1, part2])
        else:
            flat = np.ravel(np.asarray(y, dtype=float))

        if flat.shape[0] == 2*self.S + 1:
            flat = flat[:-1]  # discard last element if extra
        if flat.shape[0] != 2*self.S:
            raise ValueError(f"Expected length {2*self.S}, got {flat.shape[0]}")
        return flat

    def _derivatives(self, t, y):
        y = self._ensure_flat_y(y)
        logB = y[:self.S]
        pclock = y[self.S:2*self.S]

        B = np.exp(logB) * (~self.waiting)
        local_growth = self.r - self.C.dot(B)

        dlogB = np.zeros(self.S)
        dpclock = np.zeros(self.S)

        for i in range(self.S):
            if not self.waiting[i]:
                # Non-waiting => normal logB derivative
                dlogB[i] = local_growth[i] + np.exp(np.log(INV) - logB[i])
            else:
                # waiting => track invasion in pclock
                denom = local_growth[i] + MORTALITY_RATE
                if abs(denom) > 1e-14:
                    est_prob = local_growth[i] / denom
                else:
                    est_prob = 0.0
                dpclock[i] = INV * est_prob / BODY_MASS

        return np.concatenate([dlogB, dpclock])

    def _event_fn(self, t, y, sw):
        """
        2*S event values:
          event(2*i)   = local_growth[i]
          event(2*i+1) = pclock[i]
        """
        y = self._ensure_flat_y(y)
        logB = y[:self.S]
        pclock = y[self.S:2*self.S]

        B = np.exp(logB) * (~self.waiting)
        local_growth = self.r - self.C.dot(B)

        gvals = []
        for i in range(self.S):
            gvals.append(local_growth[i])
            gvals.append(pclock[i])
        return np.array(gvals, dtype=float)

    def _handle_event_fn(self, t, y):
        """Event logic by sign changes in _event_fn."""
        y = self._ensure_flat_y(y)
        logB = y[:self.S].copy()
        pclock = y[self.S:2*self.S].copy()

        vals = self._event_fn(t, y, None)
        current_sw = np.sign(vals)

        if self.last_sw is None:
            self.last_sw = current_sw
            return y

        changed = np.where(current_sw != self.last_sw)[0]

        B = np.exp(logB) * (~self.waiting)
        local_growth = self.r - self.C.dot(B)

        for idx in changed:
            i_species = idx // 2
            is_clock_event = (idx % 2 == 1)
            if not is_clock_event:
                # local_growth crossing zero
                if self.waiting[i_species] and local_growth[i_species] < 0:
                    self.waiting[i_species] = False
                    logB[i_species] = np.log(INV/10.0)
                elif (not self.waiting[i_species]) and (local_growth[i_species] < 0):
                    self.waiting[i_species] = True
                    pclock[i_species] = np.log(np.random.rand())
                    logB[i_species] = np.log(INV/10.0)
            else:
                # pclock crossing zero => establishment
                if self.waiting[i_species]:
                    denom = local_growth[i_species] + MORTALITY_RATE
                    est_prob = (local_growth[i_species] / denom) if abs(denom) > 1e-14 else 0.0
                    val = BODY_MASS / est_prob if est_prob > 0 else BODY_MASS
                    if val > 1:
                        logB[i_species] = 0.0
                    else:
                        logB[i_species] = np.log(val)
                    self.waiting[i_species] = False
                pclock[i_species] = np.log(np.random.rand())

        self.last_sw = current_sw
        return np.concatenate([logB, pclock])

    def run(self):
        logger.info("Starting PSD2 simulation with Assimulo...")

        # Build y0
        y0 = np.concatenate([self.logB, self.poisson_clock])
        init_vals = self._event_fn(0.0, y0, None)
        self.last_sw = np.sign(init_vals)

        # Problem setup
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0)
        problem.name = 'PSD2 Problem'
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        problem.number_of_state_events = 2*self.S

        # Solver configuration
        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        # For large S, "Dense" can be huge. Use "SPGMR" for an iterative approach.
        solver.linear_solver = 'SPGMR'
        solver.rtol = 1e-5  # Looser tolerance for big system
        solver.atol = 1e-4

        solver.options['hmin'] = 1e-4
        solver.options['maxh'] = 20
        solver.options['root_tol'] = 1e-6
        solver.options["mxhnil"] = 5
        solver.options['maxsteps'] = 300

        # Chunk the integration to avoid huge steps
        chunk_size = 2000
        times = np.arange(0, self.tmax + chunk_size, chunk_size)
        times = np.unique(np.clip(times, 0, self.tmax))

        # Recording times (for output snapshots)
        record_times = np.arange(0, self.tmax + self.record_step, self.record_step)
        record_times = np.unique(np.clip(record_times, 0, self.tmax))

        # ---- Initial storage at time 0 ----
        B0 = np.exp(self.logB) * (~self.waiting)
        self.trajectory[0, :] = B0
        self.wait_trajectory[0, :] = self.waiting.copy()
        self.time_points[0] = 0.0

        # Compute extra diagnostics at t = 0
        growth0 = self.r - self.C.dot(B0)
        inv_rate0 = np.zeros(self.S)
        est_prob0 = np.zeros(self.S)
        for i in range(self.S):
            if self.waiting[i]:
                denom = growth0[i] + MORTALITY_RATE
                est_prob0[i] = growth0[i] / denom if abs(denom) > 1e-14 else 0.0
                inv_rate0[i] = INV * est_prob0[i] / BODY_MASS
            else:
                B_val = np.exp(self.logB[i])
                inv_rate0[i] = INV / B_val if B_val > 0 else 0
                est_prob0[i] = float('nan')
        self.poisson_clock_traj[0, :] = self.poisson_clock.copy()
        self.growth_rate_traj[0, :]   = growth0
        self.invasion_rate_traj[0, :] = inv_rate0
        self.establishment_prob_traj[0, :] = est_prob0
        self.record_idx = 1
        # ------------------------------------

        rt_idx = 1
        for i in range(len(times)-1):
            t_start = times[i]
            t_end   = times[i+1]

            solver.simulate(t_end, ncp=0)

            y_sol = self._ensure_flat_y(solver.y)
            logB_sol = y_sol[:self.S]
            pclock_sol = y_sol[self.S:2*self.S]
            self.logB = logB_sol
            self.poisson_clock = pclock_sol

            logger.info(f"PSD2 chunk finished: from t={t_start} to t={t_end} (S={self.S})")
            logger.debug(f"   => logB (sample): {self.logB[:5]}...")
            logger.debug(f"   => waiting (sample): {self.waiting[:5]}...")
            logger.debug(f"   => pclock (sample): {self.poisson_clock[:5]}...")

            # Check and record at record times within the current chunk
            while rt_idx < len(record_times) and record_times[rt_idx] <= t_end:
                rec_time = record_times[rt_idx]

                # We record the state at the current solver state
                B = np.exp(self.logB) * (~self.waiting)
                self.trajectory[self.record_idx, :] = B
                self.wait_trajectory[self.record_idx, :] = self.waiting.copy()
                self.time_points[self.record_idx] = rec_time

                # ---- Compute extra diagnostics for this record ----
                growth = self.r - self.C.dot(B)
                inv_rate = np.zeros(self.S)
                est_prob = np.zeros(self.S)
                for j in range(self.S):
                    if self.waiting[j]:
                        denom = growth[j] + MORTALITY_RATE
                        est_prob[j] = growth[j] / denom if abs(denom) > 1e-14 else 0.0
                        inv_rate[j] = INV * est_prob[j] / BODY_MASS
                    else:
                        B_val = np.exp(self.logB[j])
                        inv_rate[j] = INV / B_val if B_val > 0 else 0
                        est_prob[j] = float('nan')
                self.poisson_clock_traj[self.record_idx, :] = self.poisson_clock.copy()
                self.growth_rate_traj[self.record_idx, :] = growth
                self.invasion_rate_traj[self.record_idx, :] = inv_rate
                self.establishment_prob_traj[self.record_idx, :] = est_prob
                # -----------------------------------------------------
                
                # --- Diagnostic: Mean raw Poisson clock for waiting species ---
                if np.any(self.waiting):
                    mean_poisson = np.mean(np.exp(self.poisson_clock[self.waiting]))
                    logger.info(f"At t={rec_time}, mean raw Poisson clock for waiting species: {mean_poisson:.3f}")
                else:
                    logger.info(f"At t={rec_time}, no species are in waiting state.")
                # -------------------------------------------------------------------

                logger.info(f"PSD2: recorded at t={rec_time} => record #{self.record_idx}")
                self.record_idx += 1
                rt_idx += 1

            if t_end >= self.tmax:
                break

        logger.info("PSD2 simulation completed.")
        return (
            self.time_points[:self.record_idx],
            self.trajectory[:self.record_idx, :],
            self.wait_trajectory[:self.record_idx, :],
            self.poisson_clock_traj[:self.record_idx, :],
            self.growth_rate_traj[:self.record_idx, :],
            self.invasion_rate_traj[:self.record_idx, :],
            self.establishment_prob_traj[:self.record_idx, :]
        )
