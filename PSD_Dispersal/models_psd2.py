############################################
# models_psd2.py
############################################

"""
PSD2 approach with multi‐patch dynamics, ‘chunked’ assimilation,
and unified dispersal handling. Now, after computing the biomass
B (from logB) in each derivative evaluation, we compute the 
dispersal invasion flux via a matrix‐calculation in dispersal.py.
We then add a passive dispersal term (invasion/B) and, for adult 
dispersal, subtract an extra dispersal‐away mortality.
"""

import sys
import numpy as np
import logging
from assimulo.solvers import CVode  # or use preferred solver
from assimulo.problem import Explicit_Problem
from euler_simple import EulerSimple  # fallback solver if needed

# Import necessary constants from  config file
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    TMAX,
    RECORDING_STEP_SIZE,
    NUM_PATCHES_X,
    NUM_PATCHES_Y,
    DISPERSAL_RATE,
    LONG_DISTANCE_PROB
)
# Import dispersal routine and precomputed matrix
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123,
                 dispersal_type='adult', dispersal_away_rate=None):
        """
        :param r: 1D array of intrinsic growth rates (length S).
        :param C: 2D competition matrix (SxS).
        :param tmax: maximum simulation time.
        :param record_step: time interval for recording outputs.
        :param dispersal_type: 'adult' or 'propagule' to indicate how dispersal is handled.
        :param dispersal_away_rate: optional extra mortality due to dispersal away.
                                    If None and dispersal_type=='adult', it is set from the local dispersal matrix.
        """
        np.random.seed(seed)
        self.r = np.asarray(r, dtype=float).flatten()  # shape (S,)
        self.C = np.asarray(C, dtype=float)
        self.S = len(self.r)
        
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        # Multi-patch initialization: each species is distributed over a grid of patches.
        init_biomass = BODY_MASS / 10
        # logB is now a (S, NUM_PATCHES_Y, NUM_PATCHES_X) array
        self.logB = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), np.log(init_biomass))
        self.waiting = np.ones((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        self.poisson_clock = np.log(np.random.rand(self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        # Set dispersal parameters
        self.dispersal_type = dispersal_type
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate
        else:
            # Compute dispersal-away rate from the local dispersal matrix (shape: NUM_PATCHES_Y x NUM_PATCHES_X)
            self.dispersal_away_rate = np.sum(LOCAL_DISPERSAL_MATRIX, axis=0).reshape((NUM_PATCHES_Y, NUM_PATCHES_X))

        # For storing results (trajectory arrays now have an extra spatial dimension)
        self.nrecords = max(1, self.tmax // self.record_step)
        self.trajectory = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        self.wait_trajectory = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)

        # Diagnostic outputs
        self.poisson_clock_traj = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        self.growth_rate_traj   = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        self.invasion_rate_traj = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        self.establishment_prob_traj = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        self.record_idx = 0
        self.last_sw = None

        logger.info(f"PSD2Model init: S={self.S}, tmax={self.tmax}, record_step={self.record_step}")
        logger.debug(f"Initial logB (shape {self.logB.shape}): {self.logB}")
        logger.debug(f"Initial waiting (shape {self.waiting.shape}): {self.waiting}")
        logger.debug(f"Initial PoissonClock (shape {self.poisson_clock.shape}): {self.poisson_clock}")
        logger.debug(f"Growth rates r: {self.r}")
        logger.debug(f"Competition Matrix C shape: {self.C.shape}")

    def _ensure_flat_y(self, y):
        """
        Flatten the state arrays. The state y is a concatenation of logB and pclock,
        each of which is of shape (S, NUM_PATCHES_Y, NUM_PATCHES_X). Hence the total length is 2*S*NUM_PATCHES_Y*NUM_PATCHES_X.
        """
        y = np.asarray(y, dtype=float)
        flat = y.flatten()
        expected_length = 2 * self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        if flat.shape[0] != expected_length:
            raise ValueError(f"Expected state vector of length {expected_length}, got {flat.shape[0]}")
        return flat

    def _derivatives(self, t, y, sw):
        """
        Compute the derivatives for the state vector y.
        The state consists of logB and the Poisson clock (pclock) for each species in each patch.
        We now add a unified dispersal (invasion) term computed via compute_dispersal.
        """
        total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        logB_flat = y[:total_elements]
        pclock_flat = y[total_elements: 2 * total_elements]
        logB = logB_flat.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        pclock = pclock_flat.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        
        
        B = np.exp(logB)
        # Compute the biomass for each species in each patch.
        epsilon = 1e-300
        safeB = np.maximum(B, epsilon)
        
        # Calculate growth rates for all patches at once
        # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
        B_reshaped = B.reshape(self.S, -1)
        competitive_loss = (self.C @ B_reshaped).reshape(B.shape)
        local_growth = self.r.reshape(-1, 1, 1) - competitive_loss
        diagB = np.diag(self.C).reshape(-1, 1, 1) * B
        
        # Compute invasion flux from dispersal.
        _, invasion = compute_dispersal(B)
        
        # Use raw local growth for effective growth.
        effective_growth = local_growth
        
        sw_reshaped = np.array(sw).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        # Compute derivative for logB.
        dlogB = effective_growth + (invasion / safeB)+ 2* sw_reshaped*(local_growth+diagB)
        
        # For establishment probability: include dispersal-away in effective mortality for adult dispersal.
        non_self_growth = local_growth + diagB
        sw_reshaped = np.array(sw).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        non_self_growth[~sw_reshaped] = 0
        non_self_growth[non_self_growth < 0] = 0
        
        if self.dispersal_type == 'adult':
            denom = non_self_growth + MORTALITY_RATE + self.dispersal_away_rate
        else:
            denom = non_self_growth + MORTALITY_RATE
            
        est_prob = np.divide(non_self_growth, denom, out=np.zeros_like(non_self_growth), where=denom != 0)
        dpclock = est_prob * (invasion / BODY_MASS)
        
        return np.concatenate([dlogB.flatten(), dpclock.flatten()])

    def _event_fn(self, t, y, sw):
        """
        Compute event function values.
        We use 2*(S * NUM_PATCHES_Y * NUM_PATCHES_X) event values:
          For each species in each patch:
            event( index )   = local_growth (with intraspecific competition added back)
            event( index + total_elements) = pclock
        """
        total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        logB = y[:total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        pclock = y[total_elements:2*total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        B = np.exp(logB)
        # Calculate growth rates for all patches at once
        # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
        B_reshaped = B.reshape(self.S, -1)
        local_growth = self.r.reshape(-1, 1, 1) - (self.C @ B_reshaped).reshape(B.shape)
        # Add back intraspecific effect:
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
        return np.concatenate([local_growth.flatten(), pclock.flatten()])

    def _handle_event_fn(self, solver, event_info):
        """
        Revised multi‐patch event handler.
        Handles events from changes in local growth (from logB) and Poisson clock crossings,
        applying similar logic to the original one‐patch version but extended over species and patches.
        """
        import math
        total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        # Copy the current state and reshape into multi‐patch arrays.
        y = solver.y.copy()  # current state vector (length 2*total_elements)
        logB = y[:total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X)).copy()
        pclock = y[total_elements:2*total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X)).copy()
        # Convert logB to biomass.
        B = np.exp(logB)
        
        # Compute local growth rates per patch:
        # First, reshape B for the matrix multiplication.
        B_flat = B.reshape(self.S, -1)
        local_growth = self.r.reshape(-1, 1, 1) - (self.C @ B_flat).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        # Add back the intraspecific effect (diagonal term).
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B

        # Get the event information:
        state_info = np.array(event_info[0])
        changed = np.nonzero(state_info)[0]
        current_time = solver.t  # current simulation time
        
        # Convert solver.sw into an array with multi-patch shape:
        sw = np.array(solver.sw).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        # Process each event index:
        for idx in changed:
            if idx < total_elements:
                s_idx, i_idx, j_idx = np.unravel_index(idx, (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                event_val = state_info[idx]
                bg = B[s_idx, i_idx, j_idx]
                lg = local_growth[s_idx, i_idx, j_idx]
                cur_logB = logB[s_idx, i_idx, j_idx]
                cur_pclock = pclock[s_idx, i_idx, j_idx]
                if sw[s_idx, i_idx, j_idx]:
                    if event_val == +1:
                        print(f"Local growth rate {lg} of species {s_idx} at patch ({i_idx},{j_idx}) was negative while waiting "
                              f"({cur_logB:.12f}, {cur_pclock:.12f}) at time {current_time:.8f}!")
                    else:
                        print(f"Species {s_idx} at patch ({i_idx},{j_idx}): S ({lg:.12f}) -> P at time {current_time:.12f}")
                        sw[s_idx, i_idx, j_idx] = False
                        flat_index = np.ravel_multi_index((s_idx, i_idx, j_idx), (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                        solver.y[total_elements + flat_index] = 1.0
                elif event_val == +1:
                    sw_fixed = sw.copy()
                    yd = self._derivatives(solver.t, solver.y, sw_fixed.flatten())
                    yd_logB = yd[:total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                    yd_logB[s_idx, i_idx, j_idx] = 0
                    B_patch = B[:, i_idx, j_idx]
                    yd_patch = yd_logB[:, i_idx, j_idx]
                    c = -np.sum(self.C[s_idx, :] * B_patch * yd_patch)
                    print(f"Computed c = {c:.8f} at patch ({i_idx},{j_idx}) for species {s_idx}")
                    if c <= 0:
                        print(f"Species {s_idx} at patch ({i_idx},{j_idx}): Sweep {c:.8f} is not positive! "
                              f"Try reducing 'inith' parameter.")
                        transition_to_S = True
                    else:
                        prob_to_S = np.exp(-bg / (BODY_MASS * (1 + math.sqrt(math.pi / (2 * c)) * MORTALITY_RATE)))
                        print(f"prob_to_S = {prob_to_S:.8f} at patch ({i_idx},{j_idx}) for species {s_idx}")
                        transition_to_S = (np.random.rand() <= prob_to_S)
                    if not transition_to_S:
                        print(f"Species {s_idx} at patch ({i_idx},{j_idx}): P ({lg:.8f}) -> D at time {current_time:.8f}")
                        new_val = min(0, cur_logB - np.log(1 - prob_to_S))
                        flat_index = np.ravel_multi_index((s_idx, i_idx, j_idx), (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                        solver.y[flat_index] = new_val
                    if transition_to_S:
                        print(f"Species {s_idx} at patch ({i_idx},{j_idx}): P ({lg:.8f}) -> S at time {current_time:.8f}")
                        sw[s_idx, i_idx, j_idx] = True
                        flat_index = np.ravel_multi_index((s_idx, i_idx, j_idx), (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                        solver.y[total_elements + flat_index] = np.log(np.random.rand())
                else:
                    print(f"Species {s_idx} at patch ({i_idx},{j_idx}): D -> P at time {current_time:.8f}")
            else:
                idx_adjusted = idx - total_elements
                s_idx, i_idx, j_idx = np.unravel_index(idx_adjusted, (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                event_val = state_info[idx]
                cur_pclock = pclock[s_idx, i_idx, j_idx]
                if event_val == -1:
                    print(f"Poisson Clock {cur_pclock:.8f} declined for species {s_idx} at patch ({i_idx},{j_idx}) "
                          f"(waiting: {sw[s_idx, i_idx, j_idx]}) at time {current_time:.8f}!")
                else:
                    print(f"Species {s_idx} at patch ({i_idx},{j_idx}): S ({cur_pclock:.12f}) -> D at time {current_time:.12f}")
                    lg = local_growth[s_idx, i_idx, j_idx]
                    denom = lg + MORTALITY_RATE
                    if lg >= 0 and denom > 0:
                        est_prob = lg / denom
                        val = BODY_MASS / est_prob if est_prob > 0 else BODY_MASS
                        flat_index = np.ravel_multi_index((s_idx, i_idx, j_idx), (self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
                        if val > 1:
                            solver.y[flat_index] = 0.0
                        else:
                            solver.y[flat_index] = np.log(val)
                        solver.y[total_elements + flat_index] = 1.0
                        sw[s_idx, i_idx, j_idx] = False
                    else:
                        print(f"Local growth negative when pclock triggered for species {s_idx} at patch ({i_idx},{j_idx})!")
        
        # After processing, update solver.sw to remain consistent:
        solver.sw = sw.flatten().tolist()

    
    def run(self):
        logger.info("Starting PSD2 simulation with Assimulo (multi-patch)...")
        
        # Build initial state vector y0 by flattening logB and poisson_clock.
        y0 = np.concatenate([self.logB.flatten(), self.poisson_clock.flatten()])
        
        # Problem setup
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=self.waiting.flatten())
        problem.name = 'PSD2 Multi-Patch Problem'
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        problem.number_of_state_events = 2 * self.S * NUM_PATCHES_Y * NUM_PATCHES_X

        # Solver configuration
        solver = CVode(problem)
        solver.discr = 'BDF'
        solver.iter = 'Newton'
        solver.linear_solver = 'SPGMR'
        solver.rtol = 1e-6
        solver.atol = 1e-6
        solver.inith = 1e-7
        solver.maxh = 1e10
        solver.store_event_points = False
        solver.options["mxhnil"] = 5
        solver.options['maxsteps'] = 20000
        solver.options['verbosity'] = 30

        # Chunk the integration to avoid overly long steps.
        chunk_size = 2000
        times = np.arange(0, self.tmax + chunk_size, chunk_size)
        times = np.unique(np.clip(times, 0, self.tmax))
        record_times = np.arange(0, self.tmax + self.record_step, self.record_step)
        record_times = np.unique(np.clip(record_times, 0, self.tmax))

        # Run the simulation
        t, y = solver(self.tmax, record_times.shape[0]-1)
        
        total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        
        for step in range(record_times.shape[0]):
            # Find the index corresponding to the record time
            rec_idx = np.argmin(np.abs(t - record_times[step]))
            y_rec = y[rec_idx, :]
            logB = y_rec[:total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            pclock = y_rec[total_elements:2*total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            waiting = pclock < 0
            B = np.exp(logB)
            self.trajectory[step, :, :, :] = B
            self.wait_trajectory[step, :, :, :] = waiting
            self.time_points[step] = t[rec_idx]
            
            # Calculate growth rates for all patches at once
            # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            B_reshaped = B.reshape(self.S, -1)
            local_growth = self.r.reshape(-1, 1, 1) - (self.C @ B_reshaped).reshape(B.shape)
            local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
            inv_rate = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            est_prob = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            # Compute dispersal (invasion) flux
            _, invasion = compute_dispersal(B)
            for j in range(self.S):
                # For patches where local growth is positive:
                pos_mask = local_growth[j] > 0
                denom = local_growth[j] + MORTALITY_RATE
                est_prob[j][pos_mask] = local_growth[j][pos_mask] / denom[pos_mask]
                inv_rate[j] = invasion[j] * est_prob[j] / BODY_MASS
            self.poisson_clock_traj[step, :, :, :] = pclock
            self.growth_rate_traj[step, :, :, :] = local_growth
            self.invasion_rate_traj[step, :, :, :] = inv_rate
            self.establishment_prob_traj[step, :, :, :] = est_prob
            
            if np.any(waiting):
                mean_poisson = np.mean(np.exp(pclock[waiting]))
                logger.info(f"At t={t[rec_idx]:.2f}, mean raw Poisson clock for waiting patches: {mean_poisson:.3f}")
            else:
                logger.info(f"At t={t[rec_idx]:.2f}, no patches are in waiting state.")
            logger.info(f"PSD2: recorded at t={t[rec_idx]:.2f} => record #{step}")

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

############################################
# Testing
############################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import correlate
    
    def test_psd2_model():
        """
        Comprehensive testing of the PSD2Model class.
        Tests multi-patch dispersal (via compute_dispersal injection), biomass dynamics,
        growth rates, and invasion (dispersal) flux.
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Test parameters
        S = 3  # number of species
        nsteps = 250000  # extended simulation time
        
        # Adjust growth rates and competition for dynamic behavior
        r = np.array([0.8, 0.6, 0.7])
        C = np.array([
            [0.2, 0.1, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.2]
        ])
        
        print("\nAnalytical Equilibrium Analysis:")
        try:
            C_inv = np.linalg.inv(C)
            B_eq = C_inv @ r
            print(f"Analytical equilibrium biomass: {B_eq}")
            growth_rates_eq = r - C @ B_eq
            print(f"Growth rates at equilibrium: {growth_rates_eq}")
            print(f"Max absolute growth rate at equilibrium: {np.max(np.abs(growth_rates_eq))}")
            print(f"All components positive: {np.all(B_eq > 0)}")
        except np.linalg.LinAlgError:
            print("Warning: Competition matrix is not invertible")
            B_eq = None
        
        print("\nTesting PSD2Model with dispersal injection:")
        model = PSD2Model(r=r, C=C, tmax=nsteps, record_step=10, seed=42)
        t_points, traj, wait_traj, pclock_traj, growth_traj, inv_rate_traj, estab_prob_traj = model.run()
        
        if B_eq is not None:
            # Average over time and over all patches (axes 0, 2, and 3)
            mean_biomass = np.mean(traj, axis=(0,2,3))
            rel_error = (mean_biomass - B_eq) / B_eq
            print(f"\nMean final biomass: {mean_biomass}")
            print(f"Relative error from equilibrium: {rel_error}")
        
    test_psd2_model()








