############################################
# models_psd2.py
############################################
"""
PSD2 approach with 'chunked' assimilation, iterative solver to reduce memory usage,
and more frequent logging to avoid silent long steps that risk OS kills.
Now includes multi-patch dynamics with dispersal.
precomputes dispersal matrices and integrates dispersal loss consistently.
"""

import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from assimulo.solvers import CVode, ExplicitEuler
from assimulo.problem import Explicit_Problem
from euler_simple import EulerSimple
from dispersal import compute_dispersal, generate_habitat_quality, compute_dispersal_matrix
from config_base import PSD2Config
import copy  # For deep copying config objects

from visualization import (
    plot_spatial_patterns, plot_dispersal_patterns, plot_temporal_evolution,
    plot_environmental_effects, plot_seed_size_effects, plot_species_comparison,
    plot_dispersal_analysis, plot_all_analyses
)
from real_species_vis import (
    plot_real_species_distribution, plot_dispersal_mechanisms,
    plot_habitat_suitability
)

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self, r, C, tmax=None, record_step=None, seed=123, config=None, dispersal_type='adult'):
        np.random.seed(seed)
        self.config = config if config is not None else PSD2Config()
        self.dispersal_type = dispersal_type  # 'adult' or 'propagule'

        # Force r, C to correct shapes
        self.r = np.asarray(r, dtype=float).flatten()
        self.C = np.asarray(C, dtype=float)
        self.S = len(self.r)

        self.tmax = tmax if tmax is not None else self.config.TMAX
        self.record_step = record_step if record_step is not None else self.config.RECORDING_STEP_SIZE

        # Initialize with higher biomass and spatial variation
        init_biomass = self.config.BODY_MASS / 100  # Reduced initial biomass
        self.logB = np.zeros((self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
        
        # Add spatial variation in initial conditions
        for s in range(self.S):
            for y in range(self.config.NUM_PATCHES_Y):
                for x in range(self.config.NUM_PATCHES_X):
                    # variation = 1 + 0.5 * (np.random.rand() - 0.5)
                    self.logB[s, y, x] = np.log(init_biomass)

        # Initialize state variables with bounds
        self.waiting = np.ones((self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X), dtype=bool)
        self.poisson_clock = np.log(np.random.rand(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))

        # For storing results with better organization
        self.nrecords = max(1, self.tmax // self.record_step)
        self.trajectory = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
        self.wait_trajectory = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X), dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)
        self.record_idx = 0

        # Diagnostic arrays
        self.poisson_clock_traj = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
        self.growth_rate_traj = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
        self.invasion_rate_traj = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
        self.establishment_prob_traj = np.zeros((self.nrecords + 1, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))

        # Precompute dispersal matrices for efficiency
        self.dispersal_matrices = compute_dispersal_matrix(self.config)

        # Track event sign changes
        self.last_sw = None

        logger.info(f"PSD2Model init: S={self.S}, tmax={self.tmax}, record_step={self.record_step}")
        logger.debug(f"Initial logB shape: {self.logB.shape}")
        logger.debug(f"Initial waiting shape: {self.waiting.shape}")
        logger.debug(f"Initial PoissonClock shape: {self.poisson_clock.shape}")
        logger.debug(f"Growth rates r: {self.r}")
        logger.debug(f"Competition Matrix C shape: {self.C.shape}")

    def _ensure_flat_y(self, y):
        if isinstance(y, (list, tuple)) and len(y) == 2:
            part1 = np.ravel(np.asarray(y[0], dtype=float))
            part2 = np.ravel(np.asarray(y[1], dtype=float))
            flat = np.concatenate([part1, part2])
        else:
            flat = np.ravel(np.asarray(y, dtype=float))
        if flat.shape[0] == 2*self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + 1:
            flat = flat[:-1]
        if flat.shape[0] != 2*self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X:
            raise ValueError(f"Expected length {2*self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X}, got {flat.shape[0]}")
        return flat

    def _derivatives(self, t, y, sw):
        """
        Compute the derivatives of log-biomass and the Poisson clock,
        including dispersal effects.
        
        Both adult and propagule dispersal now subtract outgoing flux 
        (losses) and add incoming flux, ensuring that losses are treated 
        in the same way as mortality.
        """
        # Reshape y into logB and pclock components
        logB = y[:self.S * self.config.NUM_PATCHES_Y * self.config.NUM_PATCHES_X].reshape(
            self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
        pclock = y[self.S * self.config.NUM_PATCHES_Y * self.config.NUM_PATCHES_X:].reshape(
            self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)

        # Ensure the switch array has the proper shape
        sw = np.array(sw).reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)

        # Compute actual biomass
        B = np.exp(logB)

        # Calculate local growth: r - C @ B (reshaped) and then add intraspecific competition.
        B_reshaped = B.reshape(self.S, -1)
        local_growth = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)

        # For patches with very low growth, set a floor (handle mortality)
        fast_dying = local_growth < (-self.config.MORTALITY_RATE)
        full_mortality = np.full_like(local_growth, self.config.MORTALITY_RATE)
        full_mortality[fast_dying] = -local_growth[fast_dying]
        local_growth[fast_dying] = -self.config.MORTALITY_RATE

        # Compute dispersal fluxes using precomputed dispersal matrices
        outgoing_flux, incoming_flux = compute_dispersal(B, self.config, self.dispersal_matrices)

        # Compute the diagonal of the competition matrix times biomass
        diagB = np.diag(self.C).reshape(-1, 1, 1) * B

        # Integrate dispersal fluxes in the same way for both dispersal types:
        dlogB = local_growth + self.config.BODY_MASS / (B + 1e-10)
        # Subtract losses (outgoing) and add gains (incoming) in one step:
        flux_ratio = (incoming_flux - outgoing_flux) / (B + 1e-10)
        dlogB += flux_ratio

        # Calculate establishment probabilities (used for event handling)
        non_self_growth = local_growth + diagB
        non_self_growth[np.invert(sw)] = 0
        non_self_growth[non_self_growth < 0] = 0
        denom = non_self_growth + self.config.MORTALITY_RATE
        est_prob = np.divide(non_self_growth, denom, where=denom != 0)
        dpclock = est_prob * (self.config.BODY_MASS / (self.config.BODY_MASS + 1e-10))

        return np.concatenate([dlogB.ravel(), dpclock.ravel()])



    def _event_fn(self, t, y, sw):
        logB = y[:self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
        pclock = y[self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X:].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
        B = np.exp(logB)
        B_reshaped = B.reshape(self.S, -1)
        local_growth = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
        return np.concatenate([local_growth.ravel(), pclock.ravel()])

    def _handle_event_fn(self, solver, event_info):
        y = solver.y
        logB = y[:self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
        pclock = y[self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X:].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
        state_info = event_info[0]
        changed = np.nonzero(state_info)[0]
        B = np.exp(logB)
        B_reshaped = B.reshape(self.S, -1)
        local_growth = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
        sw = np.array(solver.sw).reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)

        for idx in changed:
            if idx < self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X:
                i_species = idx // (self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X)
                i_patch = idx % (self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X)
                i_y = i_patch // self.config.NUM_PATCHES_X
                i_x = i_patch % self.config.NUM_PATCHES_X
                
                if sw[i_species, i_y, i_x]:
                    if state_info[idx] == +1:
                        print(f"Local growth rate {local_growth[i_species, i_y, i_x]} of {i_species} at patch ({i_y}, {i_x}) was negative while waiting!")
                    else:
                        if B[i_species, i_y, i_x] > self.config.THRESHOLD:
                            print(f"{i_species} at patch ({i_y}, {i_x}) S ({local_growth[i_species, i_y, i_x]}) -> P at {solver.t}")
                            sw[i_species, i_y, i_x] = False
                            solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch + 
                                     self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X] = 1
                elif state_info[idx] == +1:
                    sw_fixed = sw.copy()
                    yd = self._derivatives(solver.t, solver.y, sw_fixed)
                    yd = yd.reshape(2*self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
                    local_B = B[:, i_y, i_x]
                    local_dlogB = yd[:self.S, i_y, i_x]
                    c = -np.sum(self.C[i_species, :] * local_B * local_dlogB)
                    print(f"Local sweep parameter c = {c} at species {i_species}, patch ({i_y}, {i_x})")
                    if c > 0:
                        prob_to_S = np.clip(
                            np.exp(-B[i_species, i_y, i_x] / (self.config.BODY_MASS * (1 + np.sqrt(np.pi/(2*c))*self.config.MORTALITY_RATE))),
                            0, 1)
                        print(f"prob_to_S = {prob_to_S}")
                        transition_to_S = (np.random.rand(1) <= prob_to_S)
                        if not transition_to_S:
                            print(f"{i_species} at patch ({i_y}, {i_x}) P ({local_growth[i_species, i_y, i_x]}) -> D at {solver.t}")
                            solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch] = min(
                                0,
                                solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch] - np.log(1-prob_to_S))
                        if transition_to_S:
                            print(f"{i_species} at patch ({i_y}, {i_x}) P ({local_growth[i_species, i_y, i_x]}) -> S at {solver.t}")
                            sw[i_species, i_y, i_x] = True
                            solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch + 
                                     self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X] = np.log(np.random.rand())
                    else:
                        print(f"Sweep {c} is not positive at species {i_species}, patch ({i_y}, {i_x})!")
                        print("Consider checking parameter values or revising the derivative calculation.")
                else:
                    print(f"{i_species} at patch ({i_y}, {i_x}) D -> P at {solver.t}")
            else:
                i_species = (idx - self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X) // (self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X)
                i_patch = (idx - self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X) % (self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X)
                i_y = i_patch // self.config.NUM_PATCHES_X
                i_x = i_patch % self.config.NUM_PATCHES_X
                    
                if state_info[idx] == -1:
                    print(f"Poisson Clock {pclock[i_species, i_y, i_x]} declined at patch ({i_y}, {i_x})!")
                else:
                    print(f"{i_species} at patch ({i_y}, {i_x}) S ({pclock[i_species, i_y, i_x]}) -> D at {solver.t}")
                    denom = local_growth[i_species, i_y, i_x] + self.config.MORTALITY_RATE
                    if local_growth[i_species, i_y, i_x] >= 0:
                        est_prob = local_growth[i_species, i_y, i_x] / denom
                        val = self.config.BODY_MASS / est_prob if est_prob > 0 else self.config.BODY_MASS
                        if val > 1:
                            solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch] = 0.0
                        else:
                            solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch] = np.log(val)
                        solver.y[i_species*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X + i_patch + 
                                 self.S*self.config.NUM_PATCHES_Y*self.config.NUM_PATCHES_X] = 1
                        sw[i_species, i_y, i_x] = False
                    else:
                        print("Local growth negative when pclock triggered!")
            
        solver.sw = sw.ravel().tolist()

    def run(self):
        logger.info("Starting PSD2 simulation with Assimulo...")
        y0 = np.concatenate([self.logB.ravel(), self.poisson_clock.ravel()])
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=self.waiting)
        problem.name = 'PSD2 Problem'
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        problem.number_of_state_events = 2 * self.S * self.config.NUM_PATCHES_Y * self.config.NUM_PATCHES_X

        solver = EulerSimple(problem)
        solver.maxsteps = 100000  # Increased max steps
        solver.inith = 0.1        # Smaller step for stability
        solver.store_event_points = False

        record_times = np.linspace(0, self.tmax, self.nrecords + 1)

        try:
            t, y = solver(self.tmax, record_times.shape[0]-1)
            for step in range(record_times.shape[0]):
                recStep = np.argmin(np.abs(t - record_times[step]))
                rec_time = t[recStep]
                pclock = y[recStep, self.S * self.config.NUM_PATCHES_Y * self.config.NUM_PATCHES_X:].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X)
                waiting = pclock < 0
                B = np.exp(y[recStep, :self.S * self.config.NUM_PATCHES_Y * self.config.NUM_PATCHES_X].reshape(self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X))
                self.trajectory[step, :] = B
                self.wait_trajectory[step, :] = waiting
                self.time_points[step] = rec_time

                # Compute diagnostics
                B_reshaped = B.reshape(self.S, -1)
                growth = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
                growth = growth + np.diag(self.C).reshape(-1, 1, 1) * B

                inv_rate = np.zeros_like(B)
                est_prob = np.zeros_like(B)
                for j in range(self.S):
                    mask = growth[j] > 0
                    denom = growth[j] + self.config.MORTALITY_RATE
                    est_prob[j, mask] = growth[j, mask] / denom[mask]
                    inv_rate[j, mask] = self.config.BODY_MASS  # Simplified as the ratio cancels

                self.poisson_clock_traj[step, :] = pclock
                self.growth_rate_traj[step, :] = growth
                self.invasion_rate_traj[step, :] = inv_rate
                self.establishment_prob_traj[step, :] = est_prob

                if np.any(waiting):
                    mean_poisson = np.mean(np.exp(pclock[waiting]))
                    logger.info(f"At t={rec_time}, mean raw Poisson clock for waiting species: {mean_poisson:.3f}")
                else:
                    logger.info(f"At t={rec_time}, no species are in waiting state.")
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
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            raise



############################################
# Test functions for PSD2Model
############################################

def test_psd2_model():
    """Comprehensive testing with enhanced visualization."""
    S = 3
    r = np.array([1.2, 1.0, 1.1])
    C = np.array([
        [0.1, 0.05, 0.05],
        [0.05, 0.1, 0.05],
        [0.05, 0.05, 0.1]
    ])
    seed_sizes = np.array([1.0, 0.5, 0.8])
    print("\nAnalytical Equilibrium Analysis:")
    C_inv = np.linalg.inv(C)
    B_eq = C_inv @ r
    print(f"Analytical equilibrium biomass: {B_eq}")
    growth_rates_eq = r - C @ B_eq
    print(f"Growth rates at equilibrium: {growth_rates_eq}")
    config = PSD2Config()
    habitat_quality = generate_habitat_quality()
    results = {}
    for dispersal_type in ['adult', 'propagule']:
        model = PSD2Model(r=r, C=C, record_step=100, dispersal_type=dispersal_type, config=config)
        # Unpack the returned tuple; trajectory is the second element.
        _, trajectory, _, _, _, _, _ = model.run()
        results[dispersal_type] = trajectory
        # Basic analysis: compute mean biomass by averaging over spatial dimensions.
        final_state = trajectory[-1]  # shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        mean_biomass = np.mean(final_state, axis=(1,2))  # now shape (S,)
        print(f"\nAnalysis for {dispersal_type} dispersal:")
        print(f"Mean final biomass: {mean_biomass}")
        print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
        plot_all_analyses(trajectory, dispersal_type, config.WIND_DIRECTION, config.WIND_STRENGTH, habitat_quality, seed_sizes)
        plot_real_species_distribution(trajectory, f'output_{dispersal_type}')
        plot_dispersal_mechanisms(trajectory, f'output_{dispersal_type}')
        plot_habitat_suitability(trajectory, f'output_{dispersal_type}')
    return results

def validate_model_assumptions():
    """Validate key model assumptions with detailed equilibrium analysis."""
    S = 3
    r = np.array([1.2, 1.0, 1.1])
    C = np.array([
        [0.1, 0.05, 0.05],
        [0.05, 0.1, 0.05],
        [0.05, 0.05, 0.1]
    ])
    print("\nAnalytical Equilibrium Analysis:")
    C_inv = np.linalg.inv(C)
    B_eq = C_inv @ r
    print(f"Analytical equilibrium biomass: {B_eq}")
    dispersal_rates = [0.1, 0.2, 0.3]
    print("\nTesting equilibrium convergence at different dispersal rates:")
    for rate in dispersal_rates:
        config = PSD2Config()
        config.DISPERSAL_RATE = rate
        model = PSD2Model(r=r, C=C, config=config)
        _, trajectory, _, _, _, _, _ = model.run()
        final_state = trajectory[-1]
        mean_biomass = np.mean(final_state, axis=(1,2))  # shape (S,)
        growth_rates = r - C @ mean_biomass
        net_growth = mean_biomass * growth_rates
        print(f"\nDispersal rate: {rate}")
        print("Mean biomass:", mean_biomass)
        print("Relative error from equilibrium:", (mean_biomass - B_eq) / B_eq)
        print("Net growth rates (should be ≈ 0):", net_growth)
        print("Growth rates (r - C*mean_biomass):", growth_rates)
        min_biomass = np.min(final_state)
        print("Minimum biomass:", min_biomass)
    print("\nTesting continuous approximation:")
    step_sizes = [0.1, 0.05, 0.01]
    for dt in step_sizes:
        config = PSD2Config()
        config.STEP_SIZE = dt
        model = PSD2Model(r=r, C=C, config=config)
        _, trajectory, _, _, _, _, _ = model.run()
        final_state = trajectory[-1]
        mean_biomass = np.mean(final_state, axis=(1,2))
        print(f"\nStep size: {dt}")
        print("Mean biomass:", mean_biomass)
        print("Relative error from equilibrium:", (mean_biomass - B_eq) / B_eq)
    print("\nSpatial homogeneity analysis:")
    final_state = trajectory[-1]
    spatial_var = np.var(final_state, axis=(1,2))
    spatial_cv = spatial_var / np.mean(final_state, axis=(1,2))
    print("Spatial coefficient of variation:", spatial_cv)
    print("\nTesting equilibrium stability:")
    config = PSD2Config()
    model = PSD2Model(r=r, C=C, config=config)
    init_biomass = B_eq.reshape(-1, 1, 1) * (1 + 0.1 * np.random.randn(S, config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    model.N = (init_biomass / config.BODY_MASS).astype(int)
    _, trajectory, _, _, _, _, _ = model.run()
    final_state = trajectory[-1]
    mean_biomass = np.mean(final_state, axis=(1,2))
    print("Final biomass with perturbed initial conditions:", mean_biomass)
    print("Relative error from equilibrium:", (mean_biomass - B_eq) / B_eq)
    return True

if __name__ == "__main__":
    print("Running model validation...")
    validate_model_assumptions()
    print("\nRunning main simulation...")
    results = test_psd2_model()
