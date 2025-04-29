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
from typing import Optional  
from assimulo.solvers import CVode  # or use preferred solver
from assimulo.problem import Explicit_Problem
from euler_simple import EulerSimple  # fallback solver if needed
# ◀◀◀ CHANGED: import the generator signature for type hints
from environment import generate_spatial_r
import math

# Import necessary constants from  config file
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    TMAX,
    RECORDING_STEP_SIZE,
    NUM_PATCHES_X,
    NUM_PATCHES_Y,
    DISPERSAL_RATE,
    LONG_DISTANCE_PROB,
    CONNECTANCE,            # ◀◀◀
    INTERACTION_STRENGTH    # ◀◀◀
)
# Import dispersal routine and precomputed matrix
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX

logger = logging.getLogger(__name__)

class PSD2Model:
    def __init__(self,
                 r,                 # either 1D array (length S) or ignored if r_field supplied
                #  C,
                 C=None,
                 *,
                 initial_B: Optional[np.ndarray] = None,
                 initial_wait:  Optional[np.ndarray] = None,
                 initial_clock: Optional[np.ndarray] = None,  
                 r_field=None,      # ◀◀◀ CHANGED: expect spatial field of shape (S, Y, X)
                 length_scale=None, # ◀◀◀ CHANGED: for on‐the‐fly generation
                 var_r=None,        # ◀◀◀ CHANGED: for on‐the‐fly generation
                 seed_field=None,   # ◀◀◀ CHANGED: seed for generate_spatial_r
                 tmax=None,
                 record_step=None,
                 seed=123,
                 dispersal_type='propagule',
                 dispersal_away_rate=None):

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
        # self.r = np.asarray(r, dtype=float).flatten()  # shape (S,)
        # self.C = np.asarray(C, dtype=float)
        self.S = len(r)
        
        # ◀◀◀ CHANGED: build competition matrix if not provided
        if C is None:
            rng = np.random.default_rng(seed)
            C = np.eye(self.S, dtype=float)
            # off‐diagonals: nonzero with prob=CONNECTANCE
            for i in range(self.S):
                for j in range(self.S):
                    if i != j and rng.random() < CONNECTANCE:
                        # interaction strength positive means j reduces i
                        C[i,j] = INTERACTION_STRENGTH * rng.random()
        self.C = np.asarray(C, dtype=float)

        # ◀◀◀ CHANGED: build self.r_field (S, Y, X)
        if r_field is None:
            if length_scale is not None and var_r is not None:
                # generate a spatial field
                self.r_field = generate_spatial_r(
                    self.S, NUM_PATCHES_Y, NUM_PATCHES_X,
                    length_scale, r, var_r, seed=seed_field
                )
            else:
                # broadcast constant r to every patch
                self.r_field = np.broadcast_to(
                    np.asarray(r, float).reshape(self.S,1,1),
                    (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
                )
        else:
            assert r_field.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
            self.r_field = r_field

        # flatten for linear algebra: shape (S, Y*X)
        self.r_flat = self.r_field.reshape(self.S, -1)
        
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        # Multi-patch initialization: each species is distributed over a grid of patches.
        # init_biomass = BODY_MASS / 10
        # # logB is now a (S, NUM_PATCHES_Y, NUM_PATCHES_X) array
        # self.logB = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), np.log(init_biomass))
        
        if initial_B is not None:  # ← caller DID supply counts
            self.logB = np.log(np.maximum(initial_B, 1e-300))
        else:
            init_biomass = BODY_MASS / 10
            self.logB = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X),
                                np.log(init_biomass))
            

        ### No species can invade like this (at least it's super unlikely)
        # self.waiting = np.ones((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        # self.poisson_clock = np.log(np.random.rand(self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        ### Start all species at all patches in D state:
        # self.waiting = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        # self.poisson_clock = np.ones((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        
        # ---------- waiting flags --------------------------------------
        if initial_wait is not None:
            assert initial_wait.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
            self.waiting = initial_wait.copy()
        else:
            # default: all populations start in D-state
            self.waiting = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), bool)

        # ---------- Poisson clocks -------------------------------------
        if initial_clock is not None:
            assert initial_clock.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
            self.poisson_clock = initial_clock.copy()
        else:
            # fresh exp(1) draws  ( –log U with U∈(0,1) )
            self.poisson_clock = -np.log(np.random.rand(self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        # Set dispersal parameters
        self.dispersal_type = dispersal_type
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate
        else:
            self.dispersal_away_rate = \
                np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten(). \
                reshape((NUM_PATCHES_Y, NUM_PATCHES_X))
            
        # For storing results (trajectory arrays now have an extra spatial dimension)
        self.nrecords = int(max(1, self.tmax // self.record_step))
        self.trajectory = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        self.wait_trajectory = np.zeros((self.nrecords + 1, self.S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        self.time_points = np.zeros(self.nrecords + 1)

        # Diagnostic outputs
        self.poisson_clock_traj   = np.zeros_like(self.trajectory)
        self.growth_rate_traj     = np.zeros_like(self.trajectory)
        self.invasion_rate_traj   = np.zeros_like(self.trajectory)
        self.establishment_prob_traj = np.zeros_like(self.trajectory)

        self.record_idx = 0
        self.last_sw = None

        logger.info(f"PSD2Model init: S={self.S}, tmax={self.tmax}, record_step={self.record_step}")
        logger.debug(f"Initial logB (shape {self.logB.shape}): {self.logB}")
        logger.debug(f"Initial waiting (shape {self.waiting.shape}): {self.waiting}")
        logger.debug(f"Initial PoissonClock (shape {self.poisson_clock.shape}): {self.poisson_clock}")
        logger.debug(f"Growth rates r: {self.r_flat}")
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
        # total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        # logB_flat = y[:total_elements]
        # pclock_flat = y[total_elements: 2 * total_elements]
        # logB = logB_flat.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        # pclock = pclock_flat.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        
        total = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        logB = y[:total].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        pclock = y[total:2*total].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        B = np.exp(logB)
        # Compute the biomass for each species in each patch.
        epsilon = 1e-300
        safeB = np.maximum(B, epsilon)
        
        # Calculate growth rates for all patches at once
        # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
        B_reshaped = B.reshape(self.S, -1)
        competitive_loss = (self.C @ B_reshaped).reshape(B.shape)
        # local_growth = (self.r_flat[:, None].reshape(local_growth.shape)
        #                 if False else None)
        # local_growth = self.r.reshape(-1, 1, 1) - competitive_loss
        local_growth = self.r_field - competitive_loss

        if self.dispersal_type == 'adult':
            local_growth = local_growth - \
                np.broadcast_to(self.dispersal_away_rate,local_growth.shape)

        diagB = np.diag(self.C).reshape(-1, 1, 1) * B
        
        # Compute invasion flux from dispersal.
        invasion = compute_dispersal(B)
        
        # Use raw local growth for effective growth.
        effective_growth = local_growth
        
        sw_reshaped = np.array(sw).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        # Compute derivative for logB.
        
        # Note the sign change (minus) to match the one-patch formulation.
        # The mortality/dispersal-away term is now scaled by 1 so that it aligns with the lecture derivations.
        dlogB = effective_growth + (invasion / safeB) - 2 * sw_reshaped * (local_growth + diagB)
        ## With local dispersal, sudden biomass increase in
        ## neighbouring patches can lead to large invasion/safeB
        ## terms that destabilise EulerSimple.  We put a limit to the
        ## impact here.
        dlogB = np.clip(dlogB, a_min=None, a_max=10)

        
        # For establishment probability: include dispersal-away in effective mortality for adult dispersal.
        non_self_growth = local_growth + diagB
        sw_reshaped = np.array(sw).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        non_self_growth[~sw_reshaped] = 0
        non_self_growth[non_self_growth < 0] = 0
        
        if self.dispersal_type == 'adult':
            denom = non_self_growth + MORTALITY_RATE + self.dispersal_away_rate
        else:
            denom = non_self_growth + MORTALITY_RATE
            
        est_prob = non_self_growth / denom
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
        # local_growth = self.r_field.reshape(-1, 1, 1) - (self.C @B_reshaped).reshape(B.shape)
        local_growth = self.r_field - (self.C @ B_reshaped).reshape(B.shape)

        if self.dispersal_type == 'adult':
            local_growth = local_growth - \
                np.broadcast_to(self.dispersal_away_rate,local_growth.shape)


        # Add back intraspecific effect:
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
        return np.concatenate([local_growth.flatten(), pclock.flatten()])

    def _handle_event_fn(self, solver, event_info):
        """
        Revised multi‐patch event handler.
        Handles events from changes in local growth (from logB) and Poisson clock crossings,
        applying similar logic to the original one‐patch version but extended over species and patches.
        """

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
        local_growth = self.r_field - (self.C @ B_flat).reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))

        if self.dispersal_type == 'adult':
            local_growth = local_growth - \
                np.broadcast_to(self.dispersal_away_rate,local_growth.shape)

        # Add back the intraspecific effect (diagonal term).
        local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
        
        if self.dispersal_type=='adult':
            local_growth -= np.broadcast_to(self.dispersal_away_rate, local_growth.shape)
        # add back diag‐term
        local_growth += np.diag(self.C).reshape(self.S,1,1) * B

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
                    if self.dispersal_type == 'adult':
                        denom = lg + MORTALITY_RATE + self.dispersal_away_rate[i_idx, j_idx]
                    else:
                        denom = lg + MORTALITY_RATE
                    if lg >= 0: # implies denom > 0
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
    

    # def _handle_event_fn(self, solver, event_info):
    #     """
    #     **Vectorised multi‑patch event handler**

    #     Handles two families of events without explicit Python loops:

    #     ── Local‑growth events  (first S·Y·X entries)  
    #        •  S → P  (waiting → propagule)  
    #        •  P sweep test  → S  or  P → D  
    #        •  D → P

    #     ── Poisson‑clock events  (second S·Y·X entries)  
    #        •  S → D (successful establishment)  
    #        •  diagnostic clock declines

    #     The logic is mathematically identical to the previous
    #     implementation but operates on boolean masks and array algebra,
    #     so the cost is `O(S·Y·X)` regardless of how many events fire in
    #     one step.
    #     """
    #     import numpy as np
    #     total = self.S * NUM_PATCHES_Y * NUM_PATCHES_X

    #     # ------------------------------------------------------------------
    #     # Current state (reshaped) -------------------------------------------------
    #     # ------------------------------------------------------------------
    #     y_vec   = solver.y.copy()                        # (2·total,)
    #     logB    = y_vec[:total      ].reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
    #     pclock  = y_vec[ total:2*total].reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
    #     B       = np.exp(logB)

    #     # Boolean “waiting” grid ----------------------------------------------------
    #     sw = np.asarray(solver.sw, dtype=bool).reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)

    #     # ------------------------------------------------------------------
    #     # Local growth field g = r – C B  (+ intrapecific term) --------------
    #     # ------------------------------------------------------------------
    #     B_flat       = B.reshape(self.S, -1)                       # (S, Y·X)
    #     local_growth = self.r_field - (self.C @ B_flat).reshape(B.shape)
    #     if self.dispersal_type == 'adult':
    #         local_growth = local_growth - np.broadcast_to(self.dispersal_away_rate, local_growth.shape)
    #     local_growth += np.diag(self.C).reshape(-1, 1, 1) * B      # add diagonal

    #     # ------------------------------------------------------------------
    #     # Event flags -------------------------------------------------------
    #     # event_info[0] is the sign (+1 / −1) for every root
    #     # ------------------------------------------------------------------
    #     flag = np.asarray(event_info[0], dtype=int)
    #     lg_evt_flag = flag[:total     ].reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)  # local‑growth events
    #     pc_evt_flag = flag[ total:2*total].reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)  # clock events

    #     # Helper masks ------------------------------------------------------
    #     waiting      = sw
    #     not_waiting  = ~sw

    #     # ------------------------------------------------------------------
    #     # 1.  Local‑growth events (lg_evt_flag ≠ 0) -------------------------
    #     # ------------------------------------------------------------------
    #     # 1.a  ERROR: waiting &   lg_evt_flag == +1  (growth became −ve while waiting)
    #     mask_err_lg = waiting & (lg_evt_flag == +1)
    #     if np.any(mask_err_lg):
    #         idx   = np.vstack(np.nonzero(mask_err_lg)).T
    #         print(f"[PSD2] WARNING: {idx.shape[0]} patches still waiting but growth became negative!")

    #     # 1.b  S → P  (waiting & lg_evt_flag == −1)
    #     mask_S_to_P = waiting & (lg_evt_flag == -1)
    #     if np.any(mask_S_to_P):
    #         sw[mask_S_to_P]      = False
    #         pclock[mask_S_to_P]  = 1.0
    #         print(f"[PSD2] {np.count_nonzero(mask_S_to_P)} ‘S → P’ transitions.")

    #     # 1.c  P‑state sweep test (not_waiting & lg_evt_flag == +1) -------------
    #     mask_P_sweep = not_waiting & (lg_evt_flag == +1)
    #     if np.any(mask_P_sweep):
    #         # First evaluate derivatives once for the whole grid
    #         yd_full      = self._derivatives(solver.t, solver.y, sw.flatten())
    #         yd_logB      = yd_full[:total].reshape(self.S, NUM_PATCHES_Y, NUM_PATCHES_X)

    #         # Zero‑out ∂logB for own species/patch (as in scalar version)
    #         yd_logB[mask_P_sweep] = 0.0

    #         # c(s,i,j) = − Σ_m  C[s,m] · B_m · yd_logB_m
    #         prod          = B * yd_logB                                # (S,Y,X)
    #         c_grid        = -np.tensordot(self.C, prod, axes=([1], [0]))  # (S,Y,X)

    #         # Probability to remain in S after sweep
    #         eps           = 1e-300
    #         denom         = BODY_MASS * (1 + np.sqrt(np.pi / (2*np.maximum(c_grid, eps))) * MORTALITY_RATE)

    #         prob_to_S     = np.zeros_like(c_grid)
    #         pos_c_mask    = (c_grid > 0) & mask_P_sweep
    #         prob_to_S[pos_c_mask] = np.exp(-B[pos_c_mask] / denom[pos_c_mask])

    #         # Default outcome for c≤0   → deterministic S
    #         trans_to_S    = np.ones_like(prob_to_S, dtype=bool)   # default=True
    #         rnd           = np.random.rand(*prob_to_S.shape)
    #         trans_to_S[pos_c_mask] = rnd[pos_c_mask] <= prob_to_S[pos_c_mask]

    #         # ----- P → D  (failed sweep) -----------------------------------
    #         mask_P_to_D   = (~trans_to_S) & mask_P_sweep
    #         if np.any(mask_P_to_D):
    #             new_logB = np.minimum(0.0,
    #                 logB[mask_P_to_D] -
    #                 np.log1p(-prob_to_S[mask_P_to_D]))
    #             logB[mask_P_to_D] = new_logB
    #             print(f"[PSD2] {np.count_nonzero(mask_P_to_D)} ‘P → D’ transitions.")

    #         # ----- P → S  (successful sweep) -------------------------------
    #         mask_P_to_S   = trans_to_S & mask_P_sweep
    #         if np.any(mask_P_to_S):
    #             sw[mask_P_to_S]      = True
    #             pclock[mask_P_to_S]  = np.log(np.random.rand(np.count_nonzero(mask_P_to_S)))
    #             print(f"[PSD2] {np.count_nonzero(mask_P_to_S)} ‘P → S’ transitions.")

    #     # 1.d  D → P (not_waiting & lg_evt_flag == −1) ------------------------
    #     # (purely diagnostic in the original code)
    #     mask_D_to_P = not_waiting & (lg_evt_flag == -1)
    #     if np.any(mask_D_to_P):
    #         print(f"[PSD2] {np.count_nonzero(mask_D_to_P)} ‘D → P’ transitions.")

    #     # ------------------------------------------------------------------
    #     # 2. Poisson‑clock events ------------------------------------------
    #     # ------------------------------------------------------------------
    #     # 2.a  Diagnostic clock declines (pc_evt_flag == −1)
    #     mask_clock_decl = pc_evt_flag == -1
    #     if np.any(mask_clock_decl):
    #         print(f"[PSD2] {np.count_nonzero(mask_clock_decl)} Poisson clocks declined before zero.")

    #     # 2.b  S → D  (pc_evt_flag == +1)  – establishment attempt ----------
    #     mask_S_clock = (pc_evt_flag == +1)
    #     if np.any(mask_S_clock):
    #         # denominator for establishment probability
    #         denom_pc = local_growth + MORTALITY_RATE
    #         if self.dispersal_type == 'adult':
    #             denom_pc = denom_pc + np.broadcast_to(self.dispersal_away_rate, denom_pc.shape)
    #             print(f"[PSD2] {np.count_nonzero(mask_S_clock)} ‘S → D’ transitions.")

    #         # Only meaningful where local_growth ≥ 0
    #         lg_pos_mask        = (local_growth >= 0) & mask_S_clock
    #         est_prob           = np.zeros_like(local_growth)
    #         est_prob[lg_pos_mask] = local_growth[lg_pos_mask] / denom_pc[lg_pos_mask]

    #         val                = np.where(est_prob > 0, BODY_MASS / est_prob, BODY_MASS)

    #         # Update biomass (logB)
    #         set_zero_mask      = (val > 1) & mask_S_clock
    #         logB[set_zero_mask]      = 0.0
    #         logB[mask_S_clock & ~set_zero_mask] = np.log(val[mask_S_clock & ~set_zero_mask])

    #         # Reset p‑clock and leave waiting state
    #         pclock[mask_S_clock] = 1.0
    #         sw[mask_S_clock]     = False

    #     # ------------------------------------------------------------------
    #     # 3. Write‐back to solver ------------------------------------------
    #     # ------------------------------------------------------------------
    #     solver.y[:total]            = logB.ravel()
    #     solver.y[total:2*total]     = pclock.ravel()
    #     solver.sw                   = sw.ravel().tolist()
    
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
        # solver = CVode(problem)
        # solver.discr = 'BDF'
        # solver.iter = 'Newton'
        # solver.linear_solver = 'SPGMR'
        # solver.rtol = 1e-6
        # solver.atol = 1e-6
        # solver.inith = 1e-7
        # solver.maxh = 1e10
        # solver.store_event_points = False
        # solver.options["mxhnil"] = 5
        # solver.options['maxsteps'] = 20000
        # solver.options['verbosity'] = 30

        solver = EulerSimple(problem)
        solver.options['inith'] = 1
        solver.options['maxsteps'] = 10000000
        solver.store_event_points = False

        # # #### TEST FOR TECHNICAL ERRORS: ####
        # self._derivatives(0, y0, solver.sw)
        # self._event_fn(0, y0, solver.sw)
        # self._handle_event_fn(solver,np.random.rand(2,2*self.S)*0)
        # print("TESTING OK")
        # sys.exit()

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
            # local_growth = r_field.reshape(-1, 1, 1) - (self.C @ B_reshaped).reshape(B.shape)
            local_growth = self.r_field - (self.C @ B_reshaped).reshape(B.shape)
            if self.dispersal_type == 'adult':
                local_growth = local_growth - \
                    np.broadcast_to(self.dispersal_away_rate,local_growth.shape)

            local_growth = local_growth + np.diag(self.C).reshape(-1, 1, 1) * B
            inv_rate = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            est_prob = np.zeros((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            # Compute dispersal (invasion) flux
            invasion = compute_dispersal(B)
            for j in range(self.S):
                # For patches where local growth is positive:
                pos_mask = local_growth[j] > 0
                ####!!!! Axel tried to add the correct
                ####!!!! dispersal_away_rate for adult dispersal here,
                ####!!!! but failed because he did not understand how
                ####!!!! the indexing works.
                denom = local_growth[j] + MORTALITY_RATE
                if self.dispersal_type == 'adult':
                    denom = denom + self.dispersal_away_rate

                # establishment probability only where growth>0
                est = np.zeros_like(local_growth[j])
                est[pos_mask] = local_growth[j][pos_mask] / denom[pos_mask]

                # invasion flux scaled by establishment and normalized by body mass
                inv_rate[j] = invasion[j] * est / BODY_MASS
                est_prob[j] = est
                # -----------------------------------------------
                # est_prob[j][pos_mask] = local_growth[j][pos_mask] / denom[pos_mask]
                # inv_rate[j] = invasion[j] * est_prob[j] / BODY_MASS
                # -----------------------------------------------                
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
# Testing Mean and Variance in PSD2Model
############################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def test_psd2_model():
        """
        Runs the PSD2Model simulation, computes the mean and variance of biomass
        for each species over time, and compares these with the theoretical expectations.
        """
        # Set random seed for reproducibility.
        np.random.seed(42)
        
        # # Test parameters
        # S = 3  # number of species
        # nsteps = 25000  # extended simulation time
        
        # # Define growth rates and competition matrix
        # r = np.array([0.8, 0.6, 0.7])
        # C = np.array([
        #     [0.2, 0.1, 0.1],
        #     [0.1, 0.2, 0.1],
        #     [0.1, 0.1, 0.2]
        # ])
        
        # Test parameters
        S = 3  # number of species
        nsteps = 250 # shorter simulation time for testing
        r = np.array([1.0, 1.0, 1.0])
        C = np.array([
            [1.0, 1.7, 0.4],
            [0.4, 1.0, 1.7],
            [1.7, 0.4, 1.0]
        ])

        # Compute analytical equilibrium (for reference)
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
        model = PSD2Model(r=r, C=C, tmax=nsteps, record_step=10, seed=42, dispersal_type='propagule')

        # Randomise initial state a bit:
        model.logB = model.logB + 0.5*(np.random.random_sample(model.logB.shape) - 0.5)
        
        t_points, traj, wait_traj, pclock_traj, growth_traj, inv_rate_traj, estab_prob_traj = model.run()

        # Calculate mean and variance over patches at every recorded time.
        # traj is an array of shape (n_records, S, NUM_PATCHES_Y, NUM_PATCHES_X)
        mean_time_series = np.mean(traj, axis=(2, 3))  # shape: (n_records, S)
        var_time_series  = np.var(traj, axis=(2, 3))    # shape: (n_records, S)
        
        # Example: Compute final (steady-state) mean and variance for each species
        final_biomass = traj[-1, :, :, :]  # biomass at final recorded time
        mean_final = np.mean(final_biomass, axis=(1, 2))  # mean per species
        var_final  = np.var(final_biomass, axis=(1, 2))   # variance per species
        
        print("\nFinal Mean biomass for each species:", mean_final)
        print("Final Variance for each species:", var_final)
        
        # Plot the time series for the first species (Species 0)
        plt.figure()
        plt.plot(t_points, mean_time_series[:, 0], label='Mean biomass (Species 0)')
        # Plot ± one standard deviation around the mean
        std_dev_species0 = np.sqrt(var_time_series[:, 0])
        plt.fill_between(t_points,
                         mean_time_series[:, 0] - std_dev_species0,
                         mean_time_series[:, 0] + std_dev_species0,
                         color='blue', alpha=0.2, label='Std. Dev.')
        plt.xlabel("Time")
        plt.ylabel("Biomass")
        plt.title("Time series of Mean Biomass and Variance (Species 0)")
        plt.legend()
        plt.show()
        
        if B_eq is not None:
            # Compare final simulation means with the analytical equilibrium (if available)
            rel_error = (mean_final - B_eq) / B_eq
            print(f"\nRelative error from analytical equilibrium: {rel_error}")
    
    test_psd2_model()
