############################################
# community_assembler.py
############################################
from __future__ import annotations
import abc
import time
import logging
from accelerator import np 
import numpy as _real_numpy
import os 

from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD, BODY_MASS
from assembly_utils import expand_RC, prune_extinct 
from models_ibm import IBMModel
from models_psd2 import PSD2Model
import simulation_utils as sim_utils
from trait_logic import TraitManager

log = logging.getLogger(__name__)

class StepwiseAssembler(abc.ABC):
    def __init__(self, *, base_r=1.0, frac_multi=0.00, F_sat=None, max_rounds=None,
                 max_attempts=None, richness_cap=None, seed_size=1,
                 detection_threshold=None, seed=0, checkpoint_fn=None,
                 init_state=None, init_r=None, init_C=None, 
                 init_traits=None,
                 init_attempts=0, init_round=-1, **model_kw):

        self.rng = _real_numpy.random.default_rng(seed)
        self.trait_manager = TraitManager(rng=self.rng)

        self.base_r = base_r
        self.frac_multi = frac_multi # <--- Parameter controlling scaling
        self.F_sat = F_sat
        self.max_rounds = max_rounds
        self.max_attempts = max_attempts
        self.richness_cap = richness_cap
        self.seed_size = seed_size
        self.checkpoint_fn = checkpoint_fn
        self.model_kw = model_kw

        if detection_threshold is not None:
            self.detection_threshold = float(detection_threshold) * BODY_MASS
        else:
            self.detection_threshold = THRESHOLD 
        
        self.init_state = init_state
        self.init_r = init_r
        self.init_C = init_C
        self.init_traits = init_traits
        self.init_attempts = init_attempts
        self.init_round = init_round
        
        self.hist_round = []
        self.hist_attempts = []
        self.hist_gamma = []
        self.hist_established_cum = []

    def run(self):
        if (self.init_r is not None and self.init_C is not None and 
            self.init_state is not None and self.init_traits is not None):
            
            r, C, state = self.init_r, self.init_C, self.init_state
            traits = self.init_traits
            attempts = self.init_attempts
            start_round = self.init_round + 1
            log.info(f"[{self.engine_name}] Resuming assembly from round {self.init_round} with γ={len(r)}")
        else:
            r, C, state, traits = self._initialize_founder()
            attempts = 0
            start_round = 0
        
        est_cum = 0
        rounds_iter = range(start_round, start_round + (self.max_rounds if self.max_rounds is not None else 10**12))

        for rnd in rounds_iter:
            gamma_current = len(r)
            
            # <--- RESTORED SCALING LOGIC --->
            # Calculate candidates based on gamma and frac_multi
            # e.g., if gamma=100 and frac_multi=0.05, try 5 species this round.
            n_cand = max(1, int(gamma_current * self.frac_multi))
            n_cand = 10

            if (self.max_attempts and attempts >= self.max_attempts) or \
               (self.richness_cap and gamma_current >= self.richness_cap):
                log.info(f"[{self.engine_name}] Stopping due to attempt/richness cap.")
                break

            if self.F_sat is not None and attempts >= self.F_sat * gamma_current:
                log.info(f"[{self.engine_name}] Stopping due to saturation: attempts={attempts} >= {self.F_sat}*γ")
                break

            r_big, C_big, traits_big = r, C, traits
            
            for _ in range(n_cand):
                # 1. Generate Trait
                new_trait = self.trait_manager.generate_traits(1) 
                
                # 2. Get Spatial Growth Rates (1, P)
                new_r_val = self.trait_manager.get_growth_rates(new_trait)
                new_r_val = np.asarray(new_r_val, dtype=np.float32)
                
                # 3. Calculate Interactions
                curr_S = len(r_big)
                
                target_traits_row = _real_numpy.broadcast_to(sim_utils.to_host(new_trait), (curr_S,))
                source_traits_row = sim_utils.to_host(traits_big[:curr_S])
                row_vector = self.trait_manager.get_interaction_strengths(target_traits_row, source_traits_row)
                row_inds = _real_numpy.nonzero(row_vector)[0]
                row_vals = np.asarray(row_vector[row_inds], dtype=np.float32)
                
                target_traits_col = sim_utils.to_host(traits_big[:curr_S])
                source_traits_col = _real_numpy.broadcast_to(sim_utils.to_host(new_trait), (curr_S,))
                col_vector = self.trait_manager.get_interaction_strengths(target_traits_col, source_traits_col)
                col_inds = _real_numpy.nonzero(col_vector)[0]
                col_vals = np.asarray(col_vector[col_inds], dtype=np.float32)
                
                # 4. Expand
                r_big, C_big = expand_RC(r_big, C_big, new_r_val, row_inds, col_inds, row_vals, col_vals)
                traits_big = np.concatenate([traits_big, new_trait])

            state_big = self._prepare_state_for_window(state, n_cand)
            final_state = self._run_simulation_window(r_big, C_big, state_big, n_cand)
            attempts += n_cand
            
            presence_matrix = self._get_presence_matrix(final_state)
            # Check only the last n_cand rows for establishment
            estab_mask = self._check_presence(presence_matrix[-n_cand:])
            
            if estab_mask.any():
                keep_new = np.where(estab_mask)[0] + gamma_current
                keep_all = np.r_[np.arange(gamma_current), keep_new]
                r = r_big[keep_all]
                C = C_big[np.ix_(keep_all, keep_all)]
                traits = traits_big[keep_all]
                state_post_estab = self._prune_state(final_state, keep_all)
            else:
                state_post_estab = self._prune_state(final_state, np.arange(gamma_current))

            presence_residents = self._get_presence_matrix(state_post_estab)
            alive_mask = self._check_presence(presence_residents)
            
            num_established = int(estab_mask.sum())
            num_pruned = int((~alive_mask).sum())

            if not alive_mask.all():
                res_tuple = prune_extinct(alive_mask, r, C, state_post_estab)
                r, C = res_tuple[0], res_tuple[1]
                state = res_tuple[2]
                keep_inds = res_tuple[-1]
                traits = traits[keep_inds]
            else:
                state = state_post_estab
            
            gamma_new = len(r)
            log.info(f"[{self.engine_name}] Round {rnd}: est={num_established}/{n_cand}, pruned={num_pruned}, γ {gamma_current}→ {gamma_new} (thr={self.detection_threshold / BODY_MASS:.1f} inds.)")
            
            est_cum += num_established
            self.hist_round.append(int(rnd))
            self.hist_attempts.append(int(attempts))
            self.hist_gamma.append(gamma_new)
            self.hist_established_cum.append(est_cum)

            if self.checkpoint_fn:
                self.checkpoint_fn(dict(r=r, C=C, state=state, traits=traits, occ=0, attempts=attempts, gamma=gamma_new, round=rnd, thr=self.detection_threshold))
                    
        return self._package_results(r, C, state, traits)

    def _package_results(self, r, C, final_state, traits):
        extra = dict(
            detection_threshold=self.detection_threshold,
            attempts_total=self.hist_attempts[-1] if self.hist_attempts else 0,
            round_last=self.hist_round[-1] if self.hist_round else -1,
            final_traits=traits
        )
        return r, C, final_state, extra

    # ... [Abstract methods unchanged] ...
    def _check_presence(self, presence_matrix):
        if presence_matrix.ndim == 3: return presence_matrix.any(axis=(1, 2))
        elif presence_matrix.ndim == 2: return presence_matrix.any(axis=1)
        else: raise ValueError(f"Unexpected shape: {presence_matrix.shape}")

    def _get_presence_matrix(self, state: any) -> np.ndarray: pass
    def _prune_state(self, state: any, keep_mask: np.ndarray) -> any: pass
    
    @property
    @abc.abstractmethod
    def engine_name(self) -> str: pass
    
    @abc.abstractmethod
    def _initialize_founder(self) -> tuple[np.ndarray, np.ndarray, any, np.ndarray]: pass 
    @abc.abstractmethod
    def _prepare_state_for_window(self, current_state: any, n_cand: int) -> any: pass
    @abc.abstractmethod
    def _run_simulation_window(self, r_big: np.ndarray, C_big: np.ndarray, state_big: any, n_cand: int) -> any: pass


class IBMAssembler(StepwiseAssembler):
    @property
    def engine_name(self): return "IBM"

    def _initialize_founder(self):
        if os.path.exists("IBM_biomass_value.txt"): os.remove("IBM_biomass_value.txt")
        
        traits = self.trait_manager.generate_traits(1)
        r_field = self.trait_manager.get_growth_rates(traits)
        r_field = np.asarray(r_field, dtype=np.float32)
        
        r = r_field 
        C = np.array([[1.0]], dtype=np.float32)
        
        N = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=int)
        r_reshaped = r_field[0].reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
        N[0] = np.rint(np.maximum(0, r_reshaped) / BODY_MASS).astype(int)
            
        return r, C, N, traits

    def _prepare_state_for_window(self, current_N, n_cand):
        current_N = sim_utils.to_host(current_N)
        if current_N.ndim == 2: current_N = current_N[_real_numpy.newaxis, :, :]
        S_curr, Y, X = current_N.shape
        N_big = _real_numpy.zeros((S_curr + n_cand, Y, X), dtype=int)
        N_big[:S_curr] = current_N
        for i in range(n_cand):
            patches = self.rng.choice(NUM_PATCHES_Y * NUM_PATCHES_X, min(self.seed_size, NUM_PATCHES_Y * NUM_PATCHES_X), replace=False)
            N_big[S_curr + i].reshape(-1)[patches] = 1
        return N_big

    def _run_simulation_window(self, r_big, C_big, N_big, n_cand):
        dummy_r = np.ones(len(r_big)) 
        model = IBMModel(dummy_r, C_big, initial_N=N_big, r_field=r_big, **self.model_kw)
        model.run()
        return sim_utils.to_host(model.N)

    def _get_presence_matrix(self, N):
        return (N * BODY_MASS) >= self.detection_threshold

    def _prune_state(self, N, keep_mask):
        return N[keep_mask]


class PSD2Assembler(StepwiseAssembler):
    @property
    def engine_name(self): return "PSD2"
        
    def _initialize_founder(self):
        if os.path.exists("PSD2_biomass_value.txt"): os.remove("PSD2_biomass_value.txt")
        
        traits = self.trait_manager.generate_traits(1)
        r_field = self.trait_manager.get_growth_rates(traits) 
        r_field = np.asarray(r_field, dtype=np.float32)

        r = r_field
        C = np.array([[1.0]], dtype=np.float32)
        
        r_reshaped = r_field[0].reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
        B = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X))
        B[0] = r_reshaped 
        
        W = np.zeros((1, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        PC = np.ones((1, NUM_PATCHES_Y, NUM_PATCHES_X))
        
        return r, C, (B, W, PC), traits

    def _prepare_state_for_window(self, current_state, n_cand):
        B_curr, W_curr, PC_curr = [sim_utils.to_host(x) for x in current_state]
        if B_curr.ndim == 4: B_curr = B_curr[-1]
        if W_curr.ndim == 4: W_curr = W_curr[-1]
        if PC_curr.ndim == 4: PC_curr = PC_curr[-1]
        
        if B_curr.ndim == 2: B_curr = B_curr[_real_numpy.newaxis, :, :]
        if W_curr.ndim == 2: W_curr = W_curr[_real_numpy.newaxis, :, :]
        if PC_curr.ndim == 2: PC_curr = PC_curr[_real_numpy.newaxis, :, :]
        
        S_curr = len(B_curr)
        S_big = S_curr + n_cand
        
        B_big = _real_numpy.zeros((S_big, NUM_PATCHES_Y, NUM_PATCHES_X))
        B_big[:S_curr] = B_curr
        W_big = _real_numpy.ones((S_big, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=bool)
        W_big[:S_curr] = W_curr
        
        rand_vals = self.rng.random((S_big, NUM_PATCHES_Y, NUM_PATCHES_X))
        PC_big = _real_numpy.log(rand_vals)
        PC_big[:S_curr] = PC_curr
        
        for i in range(n_cand):
            patches = self.rng.choice(NUM_PATCHES_Y * NUM_PATCHES_X, min(self.seed_size, NUM_PATCHES_Y * NUM_PATCHES_X), replace=False)
            B_big[S_curr + i].reshape(-1)[patches] = BODY_MASS
            W_big[S_curr + i].reshape(-1)[patches] = 0 
            PC_big[S_curr + i].reshape(-1)[patches] = 1 
        return (B_big, W_big, PC_big)

    def _run_simulation_window(self, r_big, C_big, state_big, n_cand):        
        B_big, W_big, PC_big = state_big
        dummy_r = np.ones(len(r_big))
        
        model = PSD2Model(dummy_r, C_big, initial_B=B_big, initial_wait=W_big, initial_clock=PC_big, 
                          n_new=n_cand, r_field=r_big, **self.model_kw)
        _, B_traj, W_traj, PC_traj, *_ = model.run()
        
        b_final = sim_utils.to_host(B_traj[-1])
        w_final = sim_utils.to_host(W_traj[-1])
        pc_final = sim_utils.to_host(PC_traj[-1])
        
        S_target = len(r_big)
        def force_shape(arr):
            if arr.ndim == 4: arr = arr[-1]
            if arr.shape != (S_target, NUM_PATCHES_Y, NUM_PATCHES_X):
                try: return arr.reshape(S_target, NUM_PATCHES_Y, NUM_PATCHES_X)
                except ValueError: pass
            return arr
                
        return (force_shape(b_final), force_shape(w_final), force_shape(pc_final))

    def _get_presence_matrix(self, state):
        B, _, _ = state
        return B >= self.detection_threshold

    def _prune_state(self, state, keep_mask):
        B, W, PC = state
        return (B[keep_mask], W[keep_mask], PC[keep_mask])
