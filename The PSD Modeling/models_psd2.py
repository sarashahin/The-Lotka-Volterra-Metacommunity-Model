
# --------------------------------------------------------
############################################
# models_PSD2.py  (speed-tuned, same math)
############################################

import os as _os  # [OPT]
_os.environ.setdefault("OMP_NUM_THREADS", "4")        # [OPT]
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")   # [OPT]
_os.environ.setdefault("MKL_NUM_THREADS", "4")        # [OPT]


import numpy as np
import logging, time
from assimulo.problem import Explicit_Problem
from euler_simple import EulerSimple
from config import (
    BODY_MASS, MORTALITY_RATE,
    STEP_SIZE, RECORDING_STEP_SIZE,
    CONNECTANCE, INTERACTION_STRENGTH,
    TMAX, INV, RTOL, ATOL)

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123):
        np.random.seed(seed)
        self.runtime_seconds = None

        # --- model data -----------------------------------------------------
        self.r = np.asarray(r, dtype=np.float64).reshape(-1)
        # ### FAST: Fortran contiguous → faster BLAS gemv (C · B)
        self.C = np.asfortranarray(np.asarray(C, dtype=np.float64))
        self.S = self.r.size
        self._S2 = 2 * self.S  

        self.tmax        = float(tmax if tmax is not None else TMAX)
        self.record_step = float(record_step if record_step is not None else RECORDING_STEP_SIZE)

        # initial state
        init_biomass = BODY_MASS / 2.0
        self.logB          = np.full(self.S, np.log(init_biomass), dtype=np.float64)
        self.waiting       = np.ones(self.S, dtype=bool)
        self.poisson_clock = np.log(np.random.rand(self.S))

        # output arrays
        self.nrecords        = int(max(1, self.tmax // self.record_step))
        self.trajectory      = np.empty((self.nrecords + 1, self.S), dtype=np.float64)
        self.wait_trajectory = np.empty((self.nrecords + 1, self.S), dtype=bool)
        self.time_points     = np.empty(self.nrecords + 1, dtype=np.float64)
        self.record_idx      = 0

        # diagnostics (unchanged)
        self.poisson_clock_traj     = np.empty((self.nrecords + 1, self.S), dtype=np.float64)
        self.growth_rate_traj       = np.empty((self.nrecords + 1, self.S), dtype=np.float64)
        self.invasion_rate_traj     = np.empty((self.nrecords + 1, self.S), dtype=np.float64)
        self.establishment_prob_traj= np.empty((self.nrecords + 1, self.S), dtype=np.float64)

        # ### FAST: precompute diagonal once
        self.C_diag = self.C.diagonal().copy()

        # ### FAST: constants & reusable buffers to avoid temporaries
        self._inv_over_BM = INV / BODY_MASS
        self._buf_B        = np.empty(self.S, dtype=np.float64)
        self._buf_local    = np.empty(self.S, dtype=np.float64)
        self._buf_diagB    = np.empty(self.S, dtype=np.float64)
        self._buf_nonself  = np.empty(self.S, dtype=np.float64)
        self._buf_dpclock  = np.empty(self.S, dtype=np.float64)
        self._ydot         = np.empty(self._S2, dtype=np.float64)
        self._event_buf    = np.empty(self._S2, dtype=np.float64)
        self._mask_wait    = np.empty(self.S, dtype=bool)              # [OPT-BUF]
        self._mask_pos     = np.empty(self.S, dtype=bool)              # [OPT-BUF]

        # [OPT-CACHE] per-step cache to avoid redoing exp/logB and C·B in _event_fn
        self._cache_y_ptr  = None
        self._cache_t      = None
        self._cache_B_valid = False

        if logger.isEnabledFor(logging.INFO):                           # [OPT-LOG]
            logger.info("PSD2Model init: S=%d, tmax=%g, record_step=%g",
                        self.S, self.tmax, self.record_step)

    # ---------------- internals ----------------
    def _ensure_cache(self, t, y_first_half):
        """[OPT-CACHE] Recompute exp(logB) and r - C@B only once per solver state."""
        y_ptr = int(y_first_half.ctypes.data)
        if (y_ptr != self._cache_y_ptr) or (t != self._cache_t) or (not self._cache_B_valid):
            np.exp(y_first_half, out=self._buf_B)                # B
            self._buf_local[:] = self.r
            self._buf_local   -= self.C.dot(self._buf_B)         # r - C@B
            np.multiply(self.C_diag, self._buf_B, out=self._buf_diagB)
            self._cache_y_ptr   = y_ptr
            self._cache_t       = t
            self._cache_B_valid = True

    # ---------------- internals ----------------

    def _derivatives(self, t, y, sw):
        """
        Fill and return d/dt [logB, pclock] in self._ydot.
        Math unchanged; we just reuse work and write in-place.
        """
        logB = y[:self.S]
        self._ensure_cache(t, logB)                              # [OPT-CACHE]

        # dlogB
        dlogB = self._ydot[:self.S]
        dlogB[:] = self._buf_local                               # r - C@B

        # + INV / B
        np.reciprocal(self._buf_B, out=self._buf_dpclock)
        self._buf_dpclock *= INV
        dlogB += self._buf_dpclock

        # - 2 * max(0, (r - C@B) + diag(C)*B) for waiting species
        np.add(self._buf_local, self._buf_diagB, out=self._buf_nonself)
        self._buf_nonself *= sw
        np.clip(self._buf_nonself, 0.0, None, out=self._buf_nonself)
        dlogB -= 2.0 * self._buf_nonself

        # dpclock = est_prob * INV/BODY_MASS
        dp = self._ydot[self.S:]
        np.add(self._buf_nonself, MORTALITY_RATE, out=self._buf_dpclock)    # denom
        np.divide(self._buf_nonself, self._buf_dpclock, out=dp, where=self._buf_dpclock != 0.0)
        dp *= self._inv_over_BM

        return self._ydot

    def _event_fn(self, t, y, sw):
        logB   = y[:self.S]
        pclock = y[self.S:self._S2]
        self._ensure_cache(t, logB)                           # [OPT-CACHE]

        # local_growth = (r - C@B) + diag(C)*B, reusing buffers
        self._buf_nonself[:] = self._buf_local                # [OPT] reuse scratch
        self._buf_nonself    += self._buf_diagB               # [OPT]

        # Return a FRESH array (np.concatenate) to avoid aliasing differences
        return np.concatenate((self._buf_nonself, pclock))    # [SAFE & SAME OUTPUTS]

    def _handle_event_fn(self, solver, event_info):
            if not logger.isEnabledFor(logging.DEBUG):               # [OPT-LOG]
                return
            y = solver.y
            logB = y[:self.S].copy()
            pclock = y[self.S:self._S2].copy()
            B = np.exp(logB)
            local_growth = self.r - self.C.dot(B) + self.C_diag * B
            state_info = event_info[0]
            changed = np.nonzero(state_info)[0]
            for idx in changed:
                if idx < self.S:
                    i = idx
                    if solver.sw[i]:
                        if state_info[idx] == +1:
                            logger.debug("Local growth %g neg while waiting (%g, %g)",
                                        local_growth[i], logB[i], pclock[i])
                    elif state_info[idx] == +1:
                        sw_fixed = solver.sw.copy()
                        yd = self._derivatives(solver.t, solver.y, sw_fixed)
                        yd[i] = 0
                        c = -np.sum(self.C[i, :] * B * yd[:self.S])
                        if c <= 0:
                            transition_to_S = True
                        else:
                            prob_to_S = np.exp(-B[i] / (BODY_MASS * (1 + np.sqrt(np.pi / (2 * c)) * MORTALITY_RATE)))
                            transition_to_S = (np.random.rand(1) <= prob_to_S)
                        if not transition_to_S:
                            solver.y[i] = min(0, solver.y[i] - np.log(max(1e-300, 1 - prob_to_S)))
                        if transition_to_S:
                            solver.sw[i] = True
                            solver.y[i + self.S] = np.log(np.random.rand())
                else:
                    i = idx - self.S
                    if state_info[idx] != -1:
                        denom = local_growth[i] + MORTALITY_RATE
                        if local_growth[i] >= 0:
                            est_prob = local_growth[i] / denom
                            val = BODY_MASS / est_prob if est_prob > 0 else BODY_MASS
                            solver.y[i] = 0.0 if val > 1 else np.log(val)
                            solver.y[i + self.S] = 1
                            solver.sw[i] = False

    def run(self):
        if logger.isEnabledFor(logging.INFO):                    # [OPT-LOG]
            logger.info("Starting PSD2 simulation with Assimulo...")
        t0 = time.perf_counter()

        y0 = np.concatenate([self.logB, self.poisson_clock])
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=self.waiting)
        problem.name = 'PSD2 Problem'
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        problem.number_of_state_events = self._S2

        solver = EulerSimple(problem)
        solver.options['inith'] = 1
        solver.options['maxsteps'] = 10_000_000
        solver.store_event_points = False

        # exact grid, no arange/unique/clip overhead
        record_times = np.linspace(0.0, self.tmax, num=self.nrecords + 1, dtype=np.float64)

        # integrate
        t, y = solver(self.tmax, record_times.shape[0] - 1)

        # robust to list returns
        t = np.asarray(t, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            if y.size % self._S2 != 0:
                raise ValueError(f"PSD2: unexpected y.size={y.size}, expected multiple of {self._S2}")
            y = y.reshape(y.size // self._S2, self._S2)

        # map times (usually direct match)
        if t.shape[0] == record_times.shape[0] and np.allclose(t, record_times):
            idx_map = None
        else:
            idx_map = np.searchsorted(t, record_times, side="left")
            np.clip(idx_map, 0, t.shape[0] - 1, out=idx_map)

        # fast local aliases
        S          = self.S
        C          = self.C
        C_diag     = self.C_diag
        r          = self.r
        traj       = self.trajectory
        wait_traj  = self.wait_trajectory
        g_traj     = self.growth_rate_traj
        inv_rate   = self.invasion_rate_traj
        est_prob   = self.establishment_prob_traj
        time_pts   = self.time_points
        inv_overBM = self._inv_over_BM

        buf_B     = self._buf_B
        buf_local = self._buf_local
        buf_diagB = self._buf_diagB
        mask_wait = self._mask_wait
        mask_pos  = self._mask_pos

        for step in range(record_times.shape[0]):
            rec_idx = step if idx_map is None else int(idx_map[step])
            cur     = y[rec_idx]

            # waiting mask (reused buffer)
            pclock       = cur[S:self._S2]
            np.less(pclock, 0.0, out=mask_wait)                     # waiting = pclock < 0

            # B = exp(logB) using cached buffers for consistency
            np.exp(cur[:S], out=buf_B)

            # masked biomass row (no temporaries)
            row = traj[step]
            row.fill(0.0)
            row[~mask_wait] = buf_B[~mask_wait]

            wait_traj[step, :] = mask_wait
            time_pts[step]     = t[rec_idx]

            # growth = r - C@B + diag(C)*B
            buf_local[:] = r
            buf_local   -= C.dot(buf_B)
            np.multiply(C_diag, buf_B, out=buf_diagB)
            np.add(buf_local, buf_diagB, out=g_traj[step])

            # est_prob & inv_rate
            growth = g_traj[step]
            np.greater(growth, 0.0, out=mask_pos)
            erow = est_prob[step]; erow.fill(0.0)
            irow = inv_rate[step]; irow.fill(0.0)
            denom = growth[mask_pos] + MORTALITY_RATE
            erow[mask_pos] = growth[mask_pos] / denom
            irow[mask_pos] = inv_overBM * erow[mask_pos]

        self.runtime_seconds = time.perf_counter() - t0
        if logger.isEnabledFor(logging.INFO):                    # [OPT-LOG]
            logger.info("PSD2 simulation completed in %.3f s.", self.runtime_seconds)

        return (self.time_points,
                self.trajectory,
                self.wait_trajectory,
                self.poisson_clock_traj,
                self.growth_rate_traj,
                self.invasion_rate_traj,
                self.establishment_prob_traj)

