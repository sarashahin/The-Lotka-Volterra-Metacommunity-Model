############################################
# models_ode.py
############################################
"""
Pure ODE approach, solved with Assimulo.
dB/dt = B * [ (r - C@B) + i/B ] = B*(r - C@B) + i
OR in log form:
d(logB_i)/dt = (r - C@B) + i*exp(-logB).
We replicate the logic from the R code in a continuous sense.
"""
import numpy as np
import logging
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    STEP_SIZE,
    RECORDING_STEP_SIZE,
    TMAX, INV, RTOL, ATOL)

logger = logging.getLogger(__name__)

class ODEModel:
    """
    Pure ODE approach:
    d(logB_i)/dt = (r_i - sum_j C_ij * B_j) + i_i * exp(-logB_i)
    We'll do y = logB vector in the solver.
    """
    def __init__(self, r, C, tmax=None, record_step=None, seed=123):
        np.random.seed(seed)
        self.r = r
        self.C = C
        self.S = len(r)
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        init_biomass = BODY_MASS/10
        self.logB = np.full(self.S, np.log(init_biomass))

        # self.nrecords = self.tmax // self.record_step if self.record_step>0 else 1
        self.nrecords = int(self.tmax // self.record_step) if self.record_step > 0 else 1
        self.trajectory = np.full((self.nrecords+1, self.S), 0.0)
        self.time_points = np.zeros(self.nrecords+1)
        self.record_idx = 0

    def _deriv(self, t, logB):
        B = np.exp(logB)
        # localGrowthRate = (r - C@B)
        local_growth = self.r - self.C.dot(B)

        # --- softened invasion ----------------------------------------------
        # instead of  INV * exp(-logB)   use   INV / (B + B0)
        # B0       = BODY_MASS /100       #  tiny “seed biomass” ≈ carrying A‑x seed
        # inv_term = INV / (B + B0)   # always finite, never huge
        # --------------------------------------------------------------------
        dlogB = local_growth + np.exp(np.log(INV) - logB)
        # dlogB    = local_growth + inv_term
        # dlogB_i/dt = local_growth[i] + exp(log(INV) - logB[i])
        # Because i_i can be species-specific, we do an array of i's => let's just do INV for each species
        # dlogB = local_growth + np.exp(np.log(INV) - logB)
        logger.debug(f"[ODEModel _deriv] t={t:.2f}, sample logB={logB[:10]}, local_growth~={local_growth[:10]}")
        return dlogB

    def run(self):
        logger.info("Starting ODE simulation with Assimulo...")

        y0 = self.logB.copy()
        problem = Explicit_Problem(self._deriv, y0, 0.0)
        problem.name = 'ODEModel'

        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        solver.linear_solver = 'SPGMR'
        solver.rtol = RTOL
        solver.atol = ATOL
        # solver.options['maxsteps'] = 300

        times = np.arange(0, self.tmax+self.record_step, self.record_step, dtype=float)

        self.trajectory[0, :] = np.exp(y0)
        self.time_points[0] = 0.0
        self.record_idx = 1

        for idx in range(1, len(times)):
            t_next = times[idx]
            solver.simulate(t_next)
            y_sol = solver.y
            self.logB = y_sol

            self.trajectory[self.record_idx, :] = np.exp(self.logB)
            self.time_points[self.record_idx] = t_next
            self.record_idx += 1

            logger.info(f"ODE progress: t={t_next}, {self.record_idx}/{self.nrecords+1} recorded.")

            if t_next >= self.tmax:
                break

        logger.info("ODE simulation completed.")
        return self.time_points[:self.record_idx], self.trajectory[:self.record_idx, :]
