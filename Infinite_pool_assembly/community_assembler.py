############################################
# community_assembler.py
############################################
"""
Object-oriented framework for stepwise community assembly.

A base class `StepwiseAssembler` implements the core assembly loop, modeled
after the behavior of the original IBM assembly process. Subclasses `IBMAssembler`
and `PSD2Assembler` provide engine-specific implementations for state management
and simulation execution.
"""
from __future__ import annotations
import abc
import time
import logging
import numpy as np

# Project Module Imports
from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD, BODY_MASS
from assembly_utils import draw_interactions, expand_RC, prune_extinct
from models_ibm import IBMModel
from models_psd2 import PSD2Model
import simulation_utils as sim_utils

log = logging.getLogger(__name__)

class StepwiseAssembler(abc.ABC):
    """
    Abstract base class for running an infinite-pool stepwise community assembly.
    """
    def __init__(self, *, base_r=1.0, frac_multi=0.05, F_sat=None, max_rounds=None,
                 max_attempts=None, richness_cap=None, seed_size=5,
                 detection_threshold=None, seed=0, checkpoint_fn=None,
                 init_state=None, init_r=None, init_C=None,
                 init_attempts=0, init_round=-1, **model_kw):

        self.rng = np.random.default_rng(seed)
        self.base_r = base_r
        self.frac_multi = frac_multi
        self.F_sat = F_sat
        self.max_rounds = max_rounds
        self.max_attempts = max_attempts
        self.richness_cap = richness_cap
        self.seed_size = seed_size
        self.checkpoint_fn = checkpoint_fn
        self.model_kw = model_kw

        # --- FIX: Ensure self.detection_threshold is ALWAYS a biomass value. ---
        if detection_threshold is not None:
            # Assume a value from the CLI is an individual count; convert to biomass.
            self.detection_threshold = float(detection_threshold) * BODY_MASS
        else:
            # Use the default THRESHOLD from config.py, which is already a biomass.
            self.detection_threshold = THRESHOLD #
        
        # Resume state
        self.init_state = init_state
        self.init_r = init_r
        self.init_C = init_C
        self.init_attempts = init_attempts
        self.init_round = init_round
        
        # History buffers
        self.hist_round = []
        self.hist_attempts = []
        self.hist_gamma = []
        self.hist_alpha_bar = []
        self.hist_alpha_pats = []
        self.hist_established_cum = []

    def run(self):
        """Executes the main assembly loop."""
        
        if self.init_r is not None and self.init_C is not None and self.init_state is not None:
            r, C, state = self.init_r, self.init_C, self.init_state
            attempts = self.init_attempts
            start_round = self.init_round + 1
            log.info(f"[{self.engine_name}] Resuming assembly from round {self.init_round} with γ={len(r)}")
        else:
            r, C, state = self._initialize_founder()
            attempts = 0
            start_round = 0
        
        est_cum = 0
        _patches_hist = self._get_history_patches()
        rounds_iter = range(start_round, start_round + (self.max_rounds if self.max_rounds is not None else 10**12))

        for rnd in rounds_iter:
            gamma_current = len(r)
            n_cand = 1 + int(self.frac_multi * gamma_current)

            if (self.max_attempts and attempts >= self.max_attempts) or \
               (self.richness_cap and gamma_current >= self.richness_cap):
                log.info(f"[{self.engine_name}] Stopping due to attempt/richness cap.")
                break

            r_big, C_big = r, C
            for _ in range(n_cand):
                row, col = draw_interactions(len(r_big), rng=self.rng)
                r_big, C_big = expand_RC(r_big, C_big, self.base_r, row, col)
            
            state_big = self._prepare_state_for_window(state, n_cand)
            final_state = self._run_simulation_window(r_big, C_big, state_big)
            attempts += n_cand
            
            presence_matrix = self._get_presence_matrix(final_state)
            estab_mask = presence_matrix[-n_cand:].any(axis=(1, 2))
            
            if estab_mask.any():
                keep_new = np.where(estab_mask)[0] + gamma_current
                keep_all = np.r_[np.arange(gamma_current), keep_new]
                r = r_big[keep_all]
                C = C_big[np.ix_(keep_all, keep_all)]
                state_post_estab = self._prune_state(final_state, keep_all)
            else:
                state_post_estab = self._prune_state(final_state, np.arange(gamma_current))

            presence_residents = self._get_presence_matrix(state_post_estab)
            alive_mask = presence_residents.any(axis=(1, 2))
            
            num_established = int(estab_mask.sum())
            num_pruned = int((~alive_mask).sum())

            if not alive_mask.all():
                r, C, state = self._prune_community(alive_mask, r, C, state_post_estab)
            else:
                state = state_post_estab
            
            gamma_new = len(r)
            log.info(f"[{self.engine_name}] Round {rnd}: est={num_established}/{n_cand}, pruned={num_pruned}, γ {gamma_current}→{gamma_new} (thr={self.detection_threshold / BODY_MASS:.1f} inds.)")
            
            final_presence = self._get_presence_matrix(state)
            rich_map = final_presence.sum(axis=0)
            
            est_cum += num_established
            self.hist_round.append(int(rnd))
            self.hist_attempts.append(int(attempts))
            self.hist_gamma.append(gamma_new)
            self.hist_established_cum.append(est_cum)
            self.hist_alpha_bar.append(float(rich_map.mean()))
            self.hist_alpha_pats.append([int(rich_map[y, x]) for (y, x) in _patches_hist])

            if self.checkpoint_fn:
                occ = final_presence.sum(axis=(1, 2))
                self.checkpoint_fn(dict(r=r, C=C, state=state, occ=occ, attempts=attempts, gamma=gamma_new, round=rnd, thr=self.detection_threshold))
            
            if self.F_sat is not None and attempts >= self.F_sat * gamma_new:
                log.info(f"[{self.engine_name}] Stopping due to saturation: attempts={attempts} >= {self.F_sat}*γ")
                break
        
        return self._package_results(r, C, state)

    def _get_history_patches(self):
        patches = [(0, 0), (0, min(NUM_PATCHES_X - 1, 1)), (min(NUM_PATCHES_Y - 1, 1), 0)]
        return patches[:min(3, NUM_PATCHES_Y * NUM_PATCHES_X)]

    def _package_results(self, r, C, final_state):
        presence = self._get_presence_matrix(final_state)
        extra = dict(
            occ_counts=presence.sum(axis=(1, 2)),
            detection_threshold=self.detection_threshold,
            attempts_total=self.hist_attempts[-1] if self.hist_attempts else 0,
            round_last=self.hist_round[-1] if self.hist_round else -1,
            frac_multi=self.frac_multi, seed_size=self.seed_size,
            ASM_round=np.asarray(self.hist_round, dtype=np.int64),
            ASM_attempts=np.asarray(self.hist_attempts, dtype=np.int64),
            ASM_established=np.asarray(self.hist_established_cum, dtype=np.int64),
            ASM_gamma=np.asarray(self.hist_gamma, dtype=np.int64),
            ASM_alpha_bar=np.asarray(self.hist_alpha_bar, dtype=np.float32),
            ASM_alpha_patches=np.asarray(self.hist_alpha_pats, dtype=np.int16),
        )
        return r, C, final_state, extra

    @property
    @abc.abstractmethod
    def engine_name(self) -> str: pass

    @abc.abstractmethod
    def _initialize_founder(self) -> tuple[np.ndarray, np.ndarray, any]: pass

    @abc.abstractmethod
    def _prepare_state_for_window(self, current_state: any, n_cand: int) -> any: pass

    @abc.abstractmethod
    def _run_simulation_window(self, r_big: np.ndarray, C_big: np.ndarray, state_big: any) -> any: pass

    @abc.abstractmethod
    def _get_presence_matrix(self, state: any) -> np.ndarray: pass

    @abc.abstractmethod
    def _prune_state(self, state: any, keep_mask: np.ndarray) -> any: pass
    
    @abc.abstractmethod
    def _prune_community(self, alive_mask: np.ndarray, r: np.ndarray, C: np.ndarray, state: any) -> tuple[np.ndarray, np.ndarray, any]: pass

class IBMAssembler(StepwiseAssembler):
    @property
    def engine_name(self): return "IBM"

    def _initialize_founder(self):
        r = np.array([self.base_r])
        C = np.array([[1.0]])
        N = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=int)
        N[0, self.rng.integers(NUM_PATCHES_Y), self.rng.integers(NUM_PATCHES_X)] = 1
        return r, C, N

    def _prepare_state_for_window(self, current_N, n_cand):
        N_big = np.pad(current_N, [(0, n_cand), (0, 0), (0, 0)], constant_values=0)
        for i in range(n_cand):
            patches = self.rng.choice(NUM_PATCHES_Y * NUM_PATCHES_X, min(self.seed_size, NUM_PATCHES_Y * NUM_PATCHES_X), replace=False)
            N_big[len(current_N) + i].flat[patches] = 1
        return N_big

    def _run_simulation_window(self, r_big, C_big, N_big):
        model = IBMModel(r_big, C_big, initial_N=N_big, **self.model_kw)
        model.run()
        return sim_utils.to_host(model.N)

    def _get_presence_matrix(self, N):
        """--- FIX: Compare biomass to biomass threshold. ---"""
        return (N * BODY_MASS) >= self.detection_threshold

    def _prune_state(self, N, keep_mask):
        return N[keep_mask]

    def _prune_community(self, alive_mask, r, C, N):
        r_pruned, C_pruned, N_pruned, _ = prune_extinct(alive_mask, r, C, N)
        return r_pruned, C_pruned, N_pruned

class PSD2Assembler(StepwiseAssembler):
    @property
    def engine_name(self): return "PSD2"
        
    def _initialize_founder(self):
        r = np.array([self.base_r])
        C = np.array([[1.0]])
        B = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X))
        B[0, self.rng.integers(NUM_PATCHES_Y), self.rng.integers(NUM_PATCHES_X)] = BODY_MASS
        W = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        PC = np.ones((1, NUM_PATCHES_Y, NUM_PATCHES_X))
        return r, C, (B, W, PC)

    def _prepare_state_for_window(self, current_state, n_cand):
        B_current, W_current, PC_current = current_state
        S_current = len(B_current)
        S_big = S_current + n_cand
        
        B_big = np.zeros((S_big, NUM_PATCHES_Y, NUM_PATCHES_X))
        B_big[:S_current] = B_current
        W_big = np.zeros((S_big, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        W_big[:S_current] = W_current
        PC_big = np.ones((S_big, NUM_PATCHES_Y, NUM_PATCHES_X))
        PC_big[:S_current] = PC_current
        
        for i in range(n_cand):
            patches = self.rng.choice(NUM_PATCHES_Y * NUM_PATCHES_X, min(self.seed_size, NUM_PATCHES_Y * NUM_PATCHES_X), replace=False)
            B_big[S_current + i].flat[patches] = BODY_MASS
        return (B_big, W_big, PC_big)

    def _run_simulation_window(self, r_big, C_big, state_big):
        B_big, W_big, PC_big = state_big
        model = PSD2Model(r_big, C_big, initial_B=B_big, initial_wait=W_big, initial_clock=PC_big, **self.model_kw)
        _, B_traj, W_traj, PC_traj, *_ = model.run()
        return (sim_utils.to_host(B_traj[-1]), sim_utils.to_host(W_traj[-1]), sim_utils.to_host(PC_traj[-1]))

    def _get_presence_matrix(self, state):
        """--- FIX: Compare biomass to biomass threshold. ---"""
        B, _, _ = state
        # self.detection_threshold is already a biomass value.
        return B >= self.detection_threshold

    def _prune_state(self, state, keep_mask):
        B, W, PC = state
        return (B[keep_mask], W[keep_mask], PC[keep_mask])

    def _prune_community(self, alive_mask, r, C, state_tuple):
        B, W, PC = state_tuple
        r_p, C_p, B_p, W_p, PC_p, _ = prune_extinct(alive_mask, r, C, B, W, PC)
        return r_p, C_p, (B_p, W_p, PC_p)

