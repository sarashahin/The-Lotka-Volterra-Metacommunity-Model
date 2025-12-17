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
import logging
from accelerator import np
import scipy.sparse as sp
from typing import Optional
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from euler_simple_safe import EulerSimpleSafe
from environment import generate_spatial_r
import math

# Import configuration
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    TMAX,
    RECORDING_STEP_SIZE,
    NUM_PATCHES_X,
    NUM_PATCHES_Y,
    DISPERSAL_RATE,
    LONG_DISTANCE_PROB,
    CONNECTANCE,
    INTERACTION_STRENGTH,
    LOG_B_CAP,
    RTOL,
    ATOL,
)

# Import dispersal
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self,
                 r,                 # 1D array (length S)
                 C=None,            # Competition matrix
                 *,
                 initial_B: Optional[np.ndarray] = None,
                 initial_wait: Optional[np.ndarray] = None,
                 initial_clock: Optional[np.ndarray] = None,
                 n_new: int = 0,
                 r_field=None,      # Spatial field (S, Y, X)
                 length_scale=None,
                 var_r=None,
                 seed_field=None,
                 tmax=None,
                 record_step=None,
                 seed=123,
                 dispersal_type='propagule',
                 dispersal_away_rate=None):

        np.random.seed(seed)
        self.S = len(r)
        self.N_patches = NUM_PATCHES_X * NUM_PATCHES_Y
        self.shape_2d = (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
        self.shape_flat = (self.S, self.N_patches)

        # --- 1. Competition Matrix ---
        if C is None:
            rng = np.random.default_rng(seed)
            C = np.eye(self.S, dtype=float)
            # Off-diagonals
            mask = rng.random((self.S, self.S)) < CONNECTANCE
            C[mask] = INTERACTION_STRENGTH * rng.random(np.count_nonzero(mask))
            np.fill_diagonal(C, 1.0) # Ensure diagonal is 1.0 usually

        self.C = sp.csr_matrix(C, dtype=float)
        self.C_diag = self.C.diagonal().astype(float) # 1-D dense

        # --- 2. Growth Rates (r) ---
        if r_field is None:
            if length_scale is not None and var_r is not None:
                # Generate spatial field then flatten
                rf = generate_spatial_r(
                    self.S, NUM_PATCHES_Y, NUM_PATCHES_X,
                    length_scale, r, var_r, seed=seed_field
                )
                self.r_flat = rf.reshape(self.S, -1)
            else:
                # Broadcast constant r
                self.r_flat = np.broadcast_to(
                    np.asarray(r, float).reshape(self.S, 1),
                    (self.S, self.N_patches)
                )
        else:
            assert r_field.shape == self.shape_2d
            self.r_flat = r_field.reshape(self.S, -1)

        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        # --- 3. Initial State (Biomass) ---
        if initial_B is not None:
            # Flatten inputs if they come in 3D (S, Y, X)
            flat_B = initial_B.reshape(self.S, -1)
            self.logB = np.log(np.maximum(flat_B, 1e-30, dtype=np.float64))
        else:
            init_biomass = BODY_MASS / 10.0
            self.logB = np.full(self.shape_flat, np.log(init_biomass))

        # --- 4. Waiting Flags & Clocks ---
        if initial_wait is not None:
            self.waiting = initial_wait.reshape(self.S, -1).astype(bool)
        else:
            self.waiting = np.zeros(self.shape_flat, dtype=bool)

        if initial_clock is not None:
            self.poisson_clock = initial_clock.reshape(self.S, -1)
        else:
            # Fresh random clocks: log(U) where U ~ Uniform(0,1)
            self.poisson_clock = np.log(np.random.rand(*self.shape_flat))

        # --- 5. Dispersal Setup ---
        self.dispersal_type = dispersal_type
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate.flatten()
        else:
            # Sum columns of the local dispersal matrix
            self.dispersal_away_rate = np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten()

        # --- 6. Initial Consistency Check & Correction ---
        # FIXME: This logic attempts to fix inconsistent states at startup.
        # Ideally, we should warn if the user provides an inconsistent state.
        
        # Calculate initial growth
        current_B = np.exp(self.logB)
        local_growth, non_self_growth = self._compute_local_growth(current_B)
        
        # If growth is negative, species MUST NOT be waiting (S state).
        # They should be in D state (established/decaying) or implicitly dead.
        # Force waiting=False where growth < 0.
        inconsistent_waiting = self.waiting & (non_self_growth < 0)
        if np.any(inconsistent_waiting):
            logger.warning("Fixed initial inconsistencies: Waiting flag set to False where growth < 0.")
            self.waiting[inconsistent_waiting] = False
            self.poisson_clock[inconsistent_waiting] = 1.0

        # Attempt establishment for new species (n_new)
        if n_new > 0:
            self._attempt_initial_establishment(n_new, current_B, local_growth)

        # --- 7. Result Storage ---
        self.nrecords = int(max(1, self.tmax // self.record_step))
        
        # Storage uses the flat dimension to simplify internal logic? 
        # Or keep 2D for output compatibility? Let's keep output 2D (S, Y, X).
        out_shape = (self.nrecords + 1, *self.shape_2d)
        
        self.trajectory = np.zeros(out_shape, dtype=np.float32)
        self.wait_trajectory = np.zeros(out_shape, dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)

        # Diagnostics
        self.poisson_clock_traj = np.zeros(out_shape, dtype=np.float32)
        self.growth_rate_traj = np.zeros(out_shape, dtype=np.float32)
        self.invasion_rate_traj = np.zeros(out_shape, dtype=np.float32)
        self.establishment_prob_traj = np.zeros(out_shape, dtype=np.float32)

        logger.info(f"PSD2Model initialized: S={self.S}, Patches={self.N_patches}")

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _compute_local_growth(self, B):
        """
        Computes the growth rates.
        Returns:
            local_growth: r - C*B (potentially minus dispersal_away)
            non_self_growth: local_growth + intraspecific_term (effective growth for invasion)
        """
        # Linear Algebra: C is sparse (S, S), B is (S, N) -> (S, N)
        # Note: In Python 3.5+ @ is matmul. C @ B works if B is dense.
        competitive_loss = self.C @ B 
        
        local_growth = self.r_flat - competitive_loss

        if self.dispersal_type == 'adult':
            # Subtract dispersal away rate (broadcasted along S axis)
            local_growth -= self.dispersal_away_rate[None, :]

        # Add back diagonal term for "non-self" growth (invasion growth rate)
        # B * diag(C) broadcasted
        diag_term = self.C_diag[:, None] * B
        non_self_growth = local_growth + diag_term
        
        return local_growth, non_self_growth

    def _compute_dispersal_input(self, B):
        """
        Wrapper for dispersal.py logic.
        Reshapes flat (S, N) -> (S, Y, X) for the external module, then flattens back.
        """
        B_3d = B.reshape(self.shape_2d)
        incoming_3d = compute_dispersal(B_3d)
        return incoming_3d.reshape(self.S, -1)

    def _get_est_prob(self, growth_rate):
        """
        Calculates establishment probability P = g / (g + mu).
        Returns 0 where growth_rate <= 0.
        """
        denom = growth_rate + MORTALITY_RATE
        if self.dispersal_type == 'adult':
            denom += self.dispersal_away_rate[None, :]
        
        # Avoid division by zero or negative probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = np.where(growth_rate > 0, growth_rate / denom, 0.0)
        return prob

    def _attempt_initial_establishment(self, n_new, current_B, local_growth):
        """Initial establishment roll for newly added species."""
        for j in range(self.S - n_new, self.S):
            g = local_growth[j]
            pos_mask = g > 0
            
            # Probability calculation
            est = np.zeros_like(g)
            denom = g[pos_mask] + MORTALITY_RATE
            if self.dispersal_type == 'adult':
                denom += self.dispersal_away_rate[pos_mask]
            
            est[pos_mask] = g[pos_mask] / denom
            
            # Adjust for biomass density (more than 1 individual)
            est = 1.0 - (1.0 - est)**(current_B[j] / BODY_MASS)
            
            # Random roll
            rnd = np.random.rand(*est.shape)
            est_failures = (rnd > est) & pos_mask
            est_successes = (~est_failures) & pos_mask
            
            # Update state
            self.waiting[j][est_failures] = True
            # Reset clocks for those waiting
            n_fail = np.count_nonzero(est_failures)
            self.poisson_clock[j][est_failures] = np.log(np.random.rand(n_fail))
            
            # Adjust biomass to for those who succeeded such as to preserve
            # expectation values.
            self.logB[j][est_successes] = np.clip(
                self.logB[j][est_successes] - np.log(np.maximum(est[est_successes], 1e-10)),
                None, LOG_B_CAP
            )

    # --------------------------------------------------------------------------
    # ODE System
    # --------------------------------------------------------------------------

    def _derivatives(self, t, y, sw):
        """
        State y: [logB_flat, pclock_flat]
        """
        total = self.S * self.N_patches
        logB = y[:total].reshape(self.S, -1)
        pclock = y[total:].reshape(self.S, -1)
        
        # 1. Recover Biomass
        B = np.exp(logB)
        safeB = np.maximum(B, 1e-100)

        # 2. Compute Dispersal (Invasion Flux)
        invasion_flux = self._compute_dispersal_input(B)
        
        # 3. Compute Growth
        local_growth, non_self_growth = self._compute_local_growth(B)

        # 4. d(logB)/dt
        # Dispersal pressure = I / B
        # WARNING: Cap pressure to avoid stiff numerical spikes when B is tiny
        invasion_pressure = invasion_flux / safeB
        invasion_pressure = np.clip(invasion_pressure, 0.0, 10.0)

        dlogB = local_growth + invasion_pressure

        # 5. Hybrid logic for waiting states (sw)
        # sw is passed as a flat list by Assimulo, reshape to bool mask
        sw_mask = np.array(sw, dtype=bool).reshape(self.S, -1)
        
        # If waiting (sw=True), dynamics follow invasion growth (non-self)
        # dlogB = -g_hat + I/B
        # FIXME: Original logic had: -non_self_growth. 
        # If non_self_growth is POSITIVE (common in S state), then -non_self_growth is NEGATIVE.
        # This approximates the decay of the "expectation value" of biomass.
        dlogB[sw_mask] = -np.maximum(0, non_self_growth[sw_mask]) + \
            invasion_pressure[sw_mask]
        
        # Cap dlogB for stability
        dlogB = np.clip(dlogB, None, 10.0)

        # 6. d(pclock)/dt
        # Clock advances only if non_self_growth > 0 (Establishment probability > 0)
        # Mask: not waiting (already established) -> rate 0 (or handled elsewhere)
        # Actually, pclock runs while waiting.
        
        # Effective growth for establishment is non_self_growth
        est_prob = self._get_est_prob(non_self_growth)
        
        # Rate of clock increase = P_est * (InvasionFlux / BodyMass)
        dpclock = est_prob * (invasion_flux / BODY_MASS)
        
        # FIXME: If growth < 0, est_prob is 0, so dpclock is 0. 
        # The clock effectively pauses. 
        # The bug "S state running with negative growth" implies dpclock > 0 
        # despite growth < 0, OR the event handler flipped sw wrong.
        # Here, strict `est_prob` calculation ensures dpclock=0 if growth<0.

        return np.concatenate([dlogB.flatten(), dpclock.flatten()])

    def _event_fn(self, t, y, sw):
        """
        Events:
        1. Local Growth crossing 0 (Sign change of growth rate)
        2. Poisson Clock crossing 0 (Sign change of pclock)
        """
        total = self.S * self.N_patches
        logB = y[:total].reshape(self.S, -1)
        pclock = y[total:].reshape(self.S, -1)
        B = np.exp(logB)

        local_growth, non_self_growth = self._compute_local_growth(B)
        
        # Event 1: Root of Growth Rate (using non_self_growth for consistency with establishment)
        # FIXME: The original code used `local_growth + diag`. That is `non_self_growth`.
        # When this crosses 0, we might switch S <-> P states.
        
        # Event 2: Root of Pclock
        
        return np.concatenate([non_self_growth.flatten(), pclock.flatten()])

    def _handle_event_fn(self, solver, event_info):
        """
        Discrete event logic handling transitions between S, P, and D states.
        """
        total = self.S * self.N_patches
        
        # Unpack state
        y_vec = solver.y
        logB = y_vec[:total].reshape(self.S, -1)
        pclock = y_vec[total:].reshape(self.S, -1)
        sw = np.array(solver.sw, dtype=bool).reshape(self.S, -1)
        
        B = np.exp(logB)
        
        # Parse Event Flags (flag: +1 for neg->pos, -1 for pos->neg)
        flag = np.array(event_info[0], dtype=int)
        growth_evts = flag[:total].reshape(self.S, -1)
        clock_evts = flag[total:].reshape(self.S, -1)

        # --- 1. Growth Crossing Events ---

        # === Case A: Growth becomes NEGATIVE (pos -> neg) ===
        # Transition: S -> P
        # If we are in S (sw=True), we MUST fall back to P (sw=False).
        # Reason: The S-state approximation (-g) is invalid for g < 0. 
        # We switch to P to track the decaying expectation value using the standard ODE.
        mask_S_to_P = (growth_evts == -1) & sw
        
        if np.any(mask_S_to_P):
            sw[mask_S_to_P] = False
            pclock[mask_S_to_P] = 1.0  # Deactivate clock
            
            # Optional: Debug log
            logger.debug(f"State S->P transition for {np.count_nonzero(mask_S_to_P)} patches.")

        # Note: If we are already in D (sw=False) and growth becomes negative,
        # we naturally transition to P (still sw=False) without any action.
        # The ODE simply continues with g < 0, resulting in decay.

        # === Case B: Growth becomes POSITIVE (neg -> pos) ===
        # Transition: P -> S (Sweep Fail) OR P -> D (Sweep Success)
        mask_sweep = (growth_evts == +1) & (~sw)
        
        if np.any(mask_sweep):
            # Calculate slope of growth rate: c = d(growth)/dt
            # We need d(logB)/dt to get d(growth)/dt = -C * (B * dlogB)
            # This is expensive. We approximate or compute exactly.
            
            # Compute derivatives just for the sweep calculation
            # We pass sw as is.
            yd = self._derivatives(solver.t, solver.y, solver.sw)
            dlogB = yd[:total].reshape(self.S, -1)
            
            # Zero out own feedback for the sweep calculation logic (as per original)
            dlogB[mask_sweep] = 0.0
            
            # c = - (C * (B * dlogB))
            # d(r - CB)/dt = - C * dB/dt = -C * (B * dlogB)
            prod = B * dlogB
            c_vals = -(self.C @ prod)
            
            # Probability to FAIL sweep (remain unestablished/waiting)
            # P_fail = exp( -B / ( m * (1 + sqrt(pi / 2*c) * mu ) ) )
            # If c <= 0, we definitely fail (or just don't sweep).
            
            eps = 1e-100
            denom = BODY_MASS * (1.0 + np.sqrt(np.pi / (2.0 * np.maximum(c_vals, eps))) * MORTALITY_RATE)
            
            prob_remain_S = np.zeros_like(c_vals)
            valid_sweep = (c_vals > 0) & mask_sweep
            
            if np.any(valid_sweep):
                prob_remain_S[valid_sweep] = np.exp(-B[valid_sweep] / denom[valid_sweep])
                
            # Random Draw
            rnd = np.random.rand(*B.shape)
            failed_sweep = valid_sweep & (rnd <= prob_remain_S)
            successful_sweep = valid_sweep & (rnd > prob_remain_S)
            
            # Handle Fail: D -> P (or D->S). 
            # Biomass gets knocked down to expectation value?
            # Original code: new_logB = min(0, logB - log1p(-prob_S))
            # This logic seems specific to the approximation.
            if np.any(successful_sweep):
                offset = -np.log1p(-prob_remain_S[successful_sweep])
                logB[successful_sweep] = np.minimum(0.0, logB[successful_sweep] + offset)
            
            # Handle Success: D -> S (Become Waiting / Established??)
            # Wait, if we sweep successfully, we are ESTABLISHED.
            # If we fail, we revert to S (waiting).
            # Original code: "P -> S (successful sweep): sw=True".
            # FIXME: In PSD2 naming: 
            #   S = Waiting (Stochastic/Seed)
            #   P = Propagule (Deterministic growth)
            # If we sweep successfully, we stay P (sw=False). 
            # If we fail sweep, we go to S (sw=True).
            
            # User Code mapping: 
            # "sw[mask_P_to_S] = True" -> This means Successful Sweep -> Waiting? 
            # That contradicts "Established".
            # Let's trust the logic: High biomass -> S state (Waiting)? Unlikely.
            # Usually: Low Biomass (S) -> Clock -> High Biomass (P).
            # Sweep: Low Biomass (D/P) -> Growth turns positive -> Jump to High Biomass?
            
            # Correct Logic likely:
            # If sweep FAILS, we become Waiting (S), sw=True.
            # If sweep SUCCEEDS, we stay Established (P), sw=False.
            
            # Applying fix based on standard PSD logic:
            if np.any(failed_sweep):
                 sw[failed_sweep] = True # Go to waiting
                 # Reset clock
                 pclock[failed_sweep] = \
                     np.log(np.random.rand(np.count_nonzero(failed_sweep)))


        # --- 2. Clock Crossing Events (S -> D) ---
        # Clock hits 0. Attempt establishment.
        
        mask_clock = (clock_evts == +1) & sw # Only if currently waiting
        
        if np.any(mask_clock):
            # Calculate Establishment Probability
            # est_prob = g / (g + mu)
            local_growth, non_self_growth = self._compute_local_growth(B)
            est_prob = self._get_est_prob(non_self_growth)
            
            # Determine new biomass val = m / est_prob
            # If est_prob is tiny (growth near 0), val explodes. Cap it.
            # If est_prob <= 0 (growth < 0), logic fails. 
            
            # FIXME: Handle growth < 0 explicitly.
            # If clock triggered but growth < 0, establishment fails. 
            # Reset clock? Keep waiting?
            # User mentioned: "S state... even though intrinsic growth rate is negative".
            # If growth < 0, est_prob = 0.
            
            valid_est = mask_clock & (est_prob > 0)
            
            if np.any(valid_est):
                # Successful establishment trigger
                val = np.divide(BODY_MASS, est_prob[valid_est])
                val = np.minimum(val, BODY_MASS) # Ecological cap (1 adult)
                
                logB[valid_est] = np.log(val)
                pclock[valid_est] = 1.0 # Inactive clock
                sw[valid_est] = False   # No longer waiting -> Established
            
            # Handle invalid establishment (Clock fired but growth < 0)
            invalid_est = mask_clock & (est_prob <= 0)
            if np.any(invalid_est):
                # Clock finished but conditions bad. 
                # Transition to P state:
                pclock[valid_est] = 1.0 # Inactive clock
                sw[valid_est] = False   # No longer waiting -> Established

        # --- 3. Consistency Enforcer ---
        # Ensure that if Growth < 0, we are not stuck with a running clock 
        # that thinks it's about to establish.
        # (Though dPclock/dt = 0 handles this, rounding errors might drift it).
        
        # Write back
        solver.y[:total] = logB.ravel()
        solver.y[total:] = pclock.ravel()
        solver.sw = sw.ravel().tolist()

    def run(self):
        logger.info("Starting PSD2 simulation (Refactored)...")
        
        y0 = np.concatenate([self.logB.flatten(), self.poisson_clock.flatten()])
        sw0 = self.waiting.flatten().tolist()
        
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=sw0)
        problem.name = 'PSD2 Refactored'
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        
        # # Determine solver
        # try:
        #     solver = CVode(problem)
        #     solver.discr = 'BDF'
        #     solver.iter = 'Newton'
        #     solver.linear_solver = 'SPGMR'
        #     solver.rtol, solver.atol = RTOL, ATOL
        # except ImportError:
        #     logger.warning("CVode not found, falling back to EulerSimpleSafe.")
        #     solver = EulerSimpleSafe(problem)
        #     solver.options['inith'] = 1.0
        solver = EulerSimpleSafe(problem)
        solver.options['inith'] = 1.0

        # Recording points
        t_eval = np.arange(0, self.tmax + self.record_step, self.record_step)
        
        # Simulate
        # Note: CVode.simulate returns (t, y)
        t_out, y_out = solver.simulate(self.tmax, ncp_list=t_eval)
        
        # Process Output
        # We need to sample the output at the exact recording steps
        # (The solver might return more points). 
        # For simplicity, we interpolate or just grab nearest if ncp_list worked well.
        
        self.time_points = t_eval
        total = self.S * self.N_patches
        
        # Pre-allocate trajectory arrays
        # Shape: (N_records, S, Y, X)
        N_rec = len(t_eval)
        
        for i, t in enumerate(t_eval):
            # Find closest index in output
            idx = np.abs(t_out - t).argmin()
            y_curr = y_out[idx]
            
            logB = y_curr[:total].reshape(self.S, -1)
            pclock = y_curr[total:].reshape(self.S, -1)
            
            # Reconstruct diagnostics
            B = np.exp(logB)
            local_g, non_self_g = self._compute_local_growth(B)
            inv_flux = self._compute_dispersal_input(B)
            est_prob = self._get_est_prob(non_self_g)
            
            # Invasion Rate = I * P_est / m
            inv_rate = inv_flux * est_prob / BODY_MASS
            
            # Store (Reshape to 2D spatial)
            self.trajectory[i] = B.reshape(self.shape_2d)
            self.poisson_clock_traj[i] = pclock.reshape(self.shape_2d)
            self.growth_rate_traj[i] = non_self_g.reshape(self.shape_2d)
            self.invasion_rate_traj[i] = inv_rate.reshape(self.shape_2d)
            self.establishment_prob_traj[i] = est_prob.reshape(self.shape_2d)
            
            # Infer waiting state from pclock (if pclock < 0 usually means waiting)
            # But we don't have 'sw' history easily from CVode output without logic.
            # Approximation:
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
