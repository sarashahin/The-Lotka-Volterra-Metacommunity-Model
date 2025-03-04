
############################################
# models_psd2.py
############################################

"""
PSD2 approach with 'chunked' assimilation, iterative solver to reduce memory usage,
and more frequent logging to avoid silent long steps that risk OS kills.
"""

import sys
import numpy as np
import logging
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

logger = logging.getLogger(__name__)

def dummy_derivatives(t, y, sw):
    return(y)

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

    def _derivatives(self, t, y, sw):
        y = self._ensure_flat_y(y) # looks rather expensive to call here
        logB = y[:self.S]
        pclock = y[self.S:2*self.S]

        B = np.exp(logB) * np.invert(sw)
        local_growth = self.r - self.C.dot(B)

        dlogB = np.zeros(self.S)
        dpclock = np.zeros(self.S) ## Should not be needed

        for i in range(self.S):
            if not sw[i]:
                # Non-waiting => normal logB derivative
                dlogB[i] = local_growth[i] + np.exp(np.log(INV) - logB[i])
            # Run pclock even if not waiting, so that waiting <-> pclock < 0
            non_self_growth = local_growth[i] + B[i]*self.C[i,i]
            denom = non_self_growth + MORTALITY_RATE
            if non_self_growth >= 0: # implies denom > 0 
                est_prob = non_self_growth / denom
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
        y = self._ensure_flat_y(y) # looks rather expensive to call here
        logB = y[:self.S]
        pclock = y[self.S:2*self.S]

        B = np.exp(logB) * np.invert(sw)
        local_growth = self.r - self.C.dot(B)

        # Remove effect of intraspecific competition
        local_growth = local_growth + np.diag(self.C) * B 
        return np.concatenate([local_growth, pclock])

    def _handle_event_fn(self, solver, event_info): #(self, t, y):
        """Event logic by sign changes in _event_fn."""
        y = self._ensure_flat_y(solver.y) # perhaps not needed?
        logB = y[:self.S].copy()
        pclock = y[self.S:2*self.S].copy()

        # A value +1 in state_info indicates that the state_event
        # function crossed zero from negative to positive and a value
        # -1 indictes that the function became negative in the
        # respective component.
        state_info = event_info[0]
        changed = np.nonzero(state_info)[0]

        B = np.exp(logB) * np.invert(solver.sw)
        local_growth = self.r - self.C.dot(B)
        
        # Remove effect of intraspecific competition
        local_growth = local_growth + np.diag(self.C) * B 

        for idx in changed:
            if idx < self.S: # event with local growth rate change
                i_species = idx
                if solver.sw[i_species]: # species was waiting?
                    if state_info[idx] == +1:
                        print(f"Local growth rate {local_growth[i_species]} of {i_species} was negative while waiting ({logB[i_species]}, {pclock[i_species]})!")
                    else:
                        print(f"{i_species} S ({local_growth[i_species]}) -> P at {solver.t}")
                        solver.sw[i_species] = False
                        solver.y[i_species] = np.log(INV/100.0)
                        solver.y[i_species+self.S] = 1
                elif state_info[idx] == +1:  # species was not waiting, goes to waiting
                    print(f"{i_species} P ({local_growth[i_species]}) -> S at {solver.t}")
                    solver.sw[i_species] = True
                    solver.y[i_species+self.S] = np.log(np.random.rand())
                    solver.y[i_species] = np.log(INV/100.0) ## should not be needed
                else:
                    print(f"{i_species} D -> P at {solver.t}")
            else: # event with pclock crossing zero => establishment
                i_species = idx - self.S
                if state_info[idx] == -1:
                    print(f"Poisson Clock {pclock[i_species]} declined ({solver.sw[i_species]})!")
                else:
                    print(f"{i_species} S ({pclock[i_species]}) -> D at {solver.t}")
                    denom = local_growth[i_species] + MORTALITY_RATE
                    if local_growth[i_species] >= 0: # implies denom > 0 
                        est_prob = local_growth[i_species] / denom
                    else:
                        print("Local growth negative when pclock triggered!")
                        sys.exit()
                    val = BODY_MASS / est_prob if est_prob > 0 else BODY_MASS
                    if val > 1: ## avoid values that are too large
                        solver.y[i_species] = 0.0
                    else:
                        solver.y[i_species] = np.log(val)
                    solver.sw[i_species] = False
        
    def run(self):
        logger.info("Starting PSD2 simulation with Assimulo...")

        # Build y0
        y0 = np.concatenate([self.logB, self.poisson_clock])
        
        # Problem setup
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=self.waiting)
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
        solver.options['maxh'] = 1000
        solver.options['root_tol'] = 1e-1
        solver.options["mxhnil"] = 5
        solver.options['maxsteps'] = 300

        # #### TESTING ####
        # self._derivatives(0, y0, solver.sw)
        # self._event_fn(0, y0, solver.sw)
        # self._handle_event_fn(solver,np.random.rand(2,2*self.S)*0)
        # print("TESTING OK")
        # sys.exit()
        
        # Chunk the integration to avoid huge steps
        chunk_size = 2000
        times = np.arange(0, self.tmax + chunk_size, chunk_size)
        times = np.unique(np.clip(times, 0, self.tmax))

        # Recording times (for output snapshots)
        record_times = np.arange(0, self.tmax + self.record_step, self.record_step)
        record_times = np.unique(np.clip(record_times, 0, self.tmax))

        # Run the simulation
        t, y = solver(self.tmax, record_times.shape[0]-1)

        recSteps = np.where(np.remainder(t,self.record_step) == 0)[0]
        if recSteps.shape[0] != record_times.shape[0]:
            sys.exit("Could not localise recording time steps after simulation.")

        for step in range(recSteps.shape[0]):
            pclock = y[recSteps[step],self.S:(2*self.S)]
            waiting = pclock < 0
            B = np.exp(y[recSteps[step],0:self.S]) * np.invert(waiting)
            self.trajectory[step, :] = B
            self.wait_trajectory[step, :] = waiting
            rec_time = t[recSteps[step]]
            self.time_points[step] = rec_time
            # Compute extra diagnostics at t = 0
            growth = self.r - self.C.dot(B)
            growth = growth + np.diag(self.C) * B
            inv_rate = np.zeros(self.S)
            est_prob = np.zeros(self.S)
            for j in range(self.S):
                if growth[j] > 0:
                    denom = growth[j] + MORTALITY_RATE
                    est_prob[j] = growth[j] / denom
                    inv_rate[j] = INV * est_prob[j] / BODY_MASS
            self.poisson_clock_traj[step, :] = pclock
            self.growth_rate_traj[step, :] = growth
            self.invasion_rate_traj[step, :] = inv_rate
            self.establishment_prob_traj[step, :] = est_prob
                
            # --- Diagnostic: Mean raw Poisson clock for waiting species ---
            if np.any(waiting):
                mean_poisson = np.mean(np.exp(pclock[waiting]))
                logger.info(f"At t={rec_time}, mean raw Poisson clock for waiting species: {mean_poisson:.3f}")
            else:
                logger.info(f"At t={rec_time}, no species are in waiting state.")
                # -------------------------------------------------------------------
                logger.info(f"PSD2: recorded at t={rec_time} => record #{step}")

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
