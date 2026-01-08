############################################
# models_psd2.py
############################################

import sys
import logging
import csv
from accelerator import np, to_cpu
import numpy as _cpu_numpy 
from typing import Optional
from euler_simple_safe import Explicit_Problem, EulerSimpleSafe
from environment import generate_spatial_r
import math
from config import (
    BODY_MASS, MORTALITY_RATE, TMAX, RECORDING_STEP_SIZE,
    NUM_PATCHES_X, NUM_PATCHES_Y, DISPERSAL_RATE, LONG_DISTANCE_PROB,
    CONNECTANCE, INTERACTION_STRENGTH, LOG_B_CAP, RTOL, ATOL, STEP_SIZE,
    ECOLOGICAL_MAX_B
)
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX

logger = logging.getLogger(__name__)

SAFE_MIN_B = 1e-30 
BUFFER_SIZE = 50 

class PSD2Model:
    def __init__(self, r, C=None, *,
                 initial_B=None, initial_wait=None, initial_clock=None,
                 n_new: int = 0, r_field=None, length_scale=None, var_r=None,
                 seed_field=None, tmax=None, record_step=None, 
                 seed=None, 
                 dispersal_type='propagule', dispersal_away_rate=None):

        self.S = len(r)
        self.N_patches = NUM_PATCHES_X * NUM_PATCHES_Y
        self.shape_2d = (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
        self.shape_flat = (self.S, self.N_patches)
        
        self._cache_t = -1.0
        self._cache = {}

        if C is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random
            C = np.eye(self.S, dtype=float)
            mask = rng.random((self.S, self.S)) < CONNECTANCE
            C[mask] = INTERACTION_STRENGTH * rng.random(np.count_nonzero(mask))
            np.fill_diagonal(C, 1.0) 

        self.C = np.asarray(C, dtype=float)
        self.C_diag = self.C.diagonal().astype(float) 

        if r_field is None:
            if length_scale is not None and var_r is not None:
                rf = generate_spatial_r(self.S, NUM_PATCHES_Y, NUM_PATCHES_X, length_scale, r, var_r, seed=seed_field)
                self.r_flat = rf.reshape(self.S, -1)
            else:
                self.r_flat = np.broadcast_to(np.asarray(r, float).reshape(self.S, 1), (self.S, self.N_patches))
        else:
            self.r_flat = r_field.reshape(self.S, -1)
            
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        if initial_B is not None:
            flat_B = initial_B.reshape(self.S, -1)
            flat_B = np.asarray(flat_B, dtype=float)
            self.logB = np.log(np.maximum(flat_B, SAFE_MIN_B, dtype=np.float64))
        else:
            init_biomass = BODY_MASS / 10.0
            self.logB = np.full(self.shape_flat, np.log(init_biomass))
            
        if initial_wait is not None:
            self.waiting = np.asarray(initial_wait.reshape(self.S, -1), dtype=bool)
        else:
            self.waiting = np.zeros(self.shape_flat, dtype=bool)

        if initial_clock is not None:
            self.poisson_clock = np.asarray(initial_clock.reshape(self.S, -1), dtype=float)
        else:
            self.poisson_clock = np.log(np.random.rand(*self.shape_flat))

        self.dispersal_type = dispersal_type
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate.flatten()
        else:
            self.dispersal_away_rate = np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten()

        current_B = np.exp(self.logB)
        local_growth, non_self_growth = self._compute_local_growth(current_B)
        inconsistent_waiting = self.waiting & (non_self_growth < 0)
        self.waiting = np.where(inconsistent_waiting, False, self.waiting)
        self.poisson_clock = np.where(inconsistent_waiting, 1.0, self.poisson_clock)

        if n_new > 0:
            self._attempt_initial_establishment(n_new, current_B, local_growth)

        self.nrecords = int(max(1, self.tmax // self.record_step))
        out_shape_cpu = (self.nrecords + 1, *self.shape_2d)
        
        self.trajectory = _cpu_numpy.zeros(out_shape_cpu, dtype=_cpu_numpy.float32)
        self.wait_trajectory = _cpu_numpy.zeros(out_shape_cpu, dtype=bool)
        self.time_points = _cpu_numpy.zeros(self.nrecords + 1)
        self.poisson_clock_traj = _cpu_numpy.zeros(out_shape_cpu, dtype=_cpu_numpy.float32)
        self.growth_rate_traj = _cpu_numpy.zeros(out_shape_cpu, dtype=_cpu_numpy.float32)
        self.invasion_rate_traj = _cpu_numpy.zeros(out_shape_cpu, dtype=_cpu_numpy.float32)
        self.establishment_prob_traj = _cpu_numpy.zeros(out_shape_cpu, dtype=_cpu_numpy.float32)

        buffer_shape = (BUFFER_SIZE, *self.shape_2d)
        self.buf_B = np.zeros(buffer_shape, dtype=float)
        self.buf_W = np.zeros(buffer_shape, dtype=bool)
        self.buf_PC = np.zeros(buffer_shape, dtype=float)
        self.buf_G = np.zeros(buffer_shape, dtype=float)
        self.buf_INV = np.zeros(buffer_shape, dtype=float)
        self.buf_EST = np.zeros(buffer_shape, dtype=float)
        
        self.buffer_idx = 0
        self.global_idx = 0

        logger.info(f"PSD2Model initialized: S={self.S}, Patches={self.N_patches} (Chunked Transfer: {BUFFER_SIZE})")

    def _flush_buffer(self):
        if self.buffer_idx == 0: return
        end_idx = self.global_idx
        start_idx = self.global_idx - self.buffer_idx
        valid = slice(0, self.buffer_idx)
        
        self.trajectory[start_idx:end_idx] = to_cpu(self.buf_B[valid])
        self.wait_trajectory[start_idx:end_idx] = to_cpu(self.buf_W[valid])
        self.poisson_clock_traj[start_idx:end_idx] = to_cpu(self.buf_PC[valid])
        self.growth_rate_traj[start_idx:end_idx] = to_cpu(self.buf_G[valid])
        self.invasion_rate_traj[start_idx:end_idx] = to_cpu(self.buf_INV[valid])
        self.establishment_prob_traj[start_idx:end_idx] = to_cpu(self.buf_EST[valid])
        self.buffer_idx = 0

    def _ensure_cache(self, t, y):
        if t == self._cache_t: return
        total = self.S * self.N_patches
        logB = y[:total].reshape(self.S, -1)
        pclock = y[total:].reshape(self.S, -1)
        logB_clamped = np.minimum(logB, LOG_B_CAP)
        B = np.exp(logB_clamped)
        safeB = np.maximum(B, SAFE_MIN_B)
        local_growth, non_self_growth = self._compute_local_growth(B)
        invasion_flux = self._compute_dispersal_input(B)
        invasion_pressure = invasion_flux / safeB
        invasion_pressure = np.clip(invasion_pressure, 0.0, 10.0)
        est_prob = self._get_est_prob(non_self_growth)
        self._cache = {'B': B, 'logB': logB, 'pclock': pclock, 'local_growth': local_growth, 'non_self_growth': non_self_growth, 'invasion_flux': invasion_flux, 'invasion_pressure': invasion_pressure, 'est_prob': est_prob}
        self._cache_t = t

    def _invalidate_cache(self): self._cache_t = -1.0

    def _compute_local_growth(self, B):
        competitive_loss = self.C @ B 
        local_growth = self.r_flat - competitive_loss
        if self.dispersal_type == 'adult': local_growth -= self.dispersal_away_rate[None, :]
        diag_term = self.C_diag[:, None] * B
        non_self_growth = local_growth + diag_term
        return local_growth, non_self_growth

    def _compute_dispersal_input(self, B):
        return compute_dispersal(B.reshape(self.shape_2d)).reshape(self.S, -1)

    def _get_est_prob(self, growth_rate):
        denom = growth_rate + MORTALITY_RATE
        if self.dispersal_type == 'adult': denom += self.dispersal_away_rate[None, :]
        return np.where(growth_rate > 0, growth_rate / denom, 0.0)

    def _attempt_initial_establishment(self, n_new, current_B, local_growth):
        for j in range(self.S - n_new, self.S):
            g = local_growth[j]
            pos_mask = g > 0
            denom = g + MORTALITY_RATE
            if self.dispersal_type == 'adult': denom += self.dispersal_away_rate
            valid_est = np.divide(g, denom)
            est = np.where(pos_mask, 1.0 - (1.0 - valid_est)**(current_B[j] / BODY_MASS), 0.0)
            rnd = np.random.rand(*est.shape)
            est_failures = (rnd > est) & pos_mask
            est_successes = (~est_failures) & pos_mask
            self.waiting[j] = np.where(est_failures, True, self.waiting[j])
            new_clocks = np.log(np.random.rand(*self.poisson_clock[j].shape))
            self.poisson_clock[j] = np.where(est_failures, new_clocks, self.poisson_clock[j])
            new_logB = self.logB[j] - np.log(np.maximum(est, 1e-10))
            self.logB[j] = np.where(est_successes, np.clip(new_logB, None, LOG_B_CAP), self.logB[j])

    def _derivatives(self, t, y, sw):
        self._ensure_cache(t, y)
        c = self._cache
        dlogB = c['local_growth'] + c['invasion_pressure']
        sw_mask = sw.reshape(self.S, -1)
        dlogB_sw = -np.maximum(0, c['non_self_growth']) + c['invasion_pressure']
        dlogB = np.where(sw_mask, dlogB_sw, dlogB)
        dlogB = np.clip(dlogB, None, 10.0)
        dpclock = c['est_prob'] * (c['invasion_flux'] / BODY_MASS)
        return np.concatenate([dlogB.flatten(), dpclock.flatten()])

    def _event_fn(self, t, y, sw):
        self._ensure_cache(t, y)
        c = self._cache
        return np.concatenate([c['non_self_growth'].flatten(), c['pclock'].flatten()])

    def _handle_event_fn(self, solver, event_info):

        ## Abbreviations and reshaping
        total = self.S * self.N_patches
        self._ensure_cache(solver.t, solver.y)
        c = self._cache
        logB, pclock, B = c['logB'], c['pclock'], c['B']
        sw = solver.sw.reshape(self.S, -1)
        rootsfound = np.asarray(event_info[0], dtype=int)
        growth_evts = rootsfound[:total].reshape(self.S, -1)
        clock_evts = rootsfound[total:].reshape(self.S, -1)
        state_modified = False

        ## S -> P transitions
        mask_S_to_P = (growth_evts == -1) & sw
        sw = np.where(mask_S_to_P, False, sw)
        pclock = np.where(mask_S_to_P, 1.0, pclock)

        ## P -> S or D transitions
        # Compute derivatives of non-self growth rate:
        mask_sweep = (growth_evts == 1) & (~sw)
        yd = self._derivatives(solver.t, solver.y, solver.sw)
        dlogB_deriv = yd[:total].reshape(self.S, -1)
        # Shortcut here to get NON-SELF rate changes,
        prod = B * np.where(mask_sweep, 0.0, dlogB_deriv)
        # c_vals are rates of change of non-self rate
        c_vals = -(self.C @ prod)
        # Remove negative sweeps (should not occur)
        valid_sweep = (c_vals > 0) & mask_sweep
        c_safe = np.where(valid_sweep, c_vals, 0.0)
        sqrt_c = np.sqrt(c_safe)
        # Compute P -> S probabilities:
        const_term = MORTALITY_RATE * np.sqrt(np.pi / 2.0)
        numerator = B * sqrt_c
        denominator = BODY_MASS * (sqrt_c + const_term)
        term = numerator / denominator
        prob_remain_S = np.where(valid_sweep, np.exp(-term), 0.0)
        # Sample P -> S and P -> D cases
        rnd = np.random.rand(*B.shape)
        failed_sweep = valid_sweep & (rnd <= prob_remain_S)
        successful_sweep = valid_sweep & (rnd > prob_remain_S)
        # Handle P -> D cases
        offset = -np.log1p(-prob_remain_S)
        # Should impose a more logical upper bound
        new_logB = np.minimum(0.0, logB + offset)
        logB = np.where(successful_sweep, new_logB, logB)
        if np.any(successful_sweep): state_modified = True
        # Handle P -> S cases
        # ??? DOESN'T THIS ALSO MODIFY STATE?
        sw = np.where(failed_sweep, True, sw)
        new_clocks = np.log(np.random.rand(*B.shape))
        pclock = np.where(failed_sweep, new_clocks, pclock)

        ## S -> D transitions
        mask_clock = (clock_evts == 1) & sw
        est_prob = c['est_prob'] 
        valid_est = mask_clock & (est_prob > 0)
        val = np.divide(BODY_MASS, est_prob + 1e-20)
        val = np.minimum(val, ECOLOGICAL_MAX_B)
        logB = np.where(valid_est, np.log(val), logB)
        if np.any(valid_est): state_modified = True
        pclock = np.where(valid_est, 1.0, pclock)
        sw = np.where(valid_est, False, sw)
        # Rare cases that SHOULD NOT OCCUR:
        invalid_est = mask_clock & (est_prob <= 0)
        pclock = np.where(invalid_est, 1.0, pclock)
        sw = np.where(invalid_est, False, sw)

        ## Write resutls back to solver object
        solver.y[:total] = logB.ravel()
        solver.y[total:] = pclock.ravel()
        solver.sw = sw.ravel()
        
        # Correct state if modified:
        if state_modified:
             new_logB = logB
             # THERE ARE TOO MANY DIFFERNT CAPS ON B!?
             logB_clamped = np.minimum(new_logB, LOG_B_CAP)
             B_new = np.exp(logB_clamped)
             local_growth, non_self_growth = self._compute_local_growth(B_new)
             c['B'] = B_new; c['logB'] = new_logB.copy(); c['local_growth'] = local_growth; c['non_self_growth'] = non_self_growth
             c['invasion_flux'] = self._compute_dispersal_input(B_new)
             safeB = np.maximum(B_new, SAFE_MIN_B)
             c['invasion_pressure'] = np.clip(c['invasion_flux'] / safeB, 0.0, 10.0)
             c['est_prob'] = self._get_est_prob(non_self_growth)
        c['pclock'] = pclock.copy() 
        return solver.y
    
    def run(self):
        logger.info("Starting PSD2 simulation (Optimized Chunked Transfer)...")
        y0 = np.concatenate([self.logB.flatten(), self.poisson_clock.flatten()])
        sw0 = self.waiting.flatten() 
        problem = Explicit_Problem(self._derivatives, y0, t0=0.0, sw0=sw0)
        problem.state_events = self._event_fn
        problem.handle_event = self._handle_event_fn
        solver = EulerSimpleSafe(problem)
        solver.options['inith'] = STEP_SIZE
        t_eval_host = _cpu_numpy.arange(0, self.tmax + self.record_step, self.record_step)
        t_out, y_out = solver.simulate(self.tmax, ncp_list=t_eval_host)
        
        total = self.S * self.N_patches
        for i, t in enumerate(t_eval_host):
            idx = _cpu_numpy.abs(t_out - t).argmin()
            y_curr = y_out[idx]
            logB = y_curr[:total].reshape(self.S, -1)
            pclock = y_curr[total:].reshape(self.S, -1)
            logB_clamped = np.minimum(logB, LOG_B_CAP)
            B = np.exp(logB_clamped)
            local_g, non_self_g = self._compute_local_growth(B)
            inv_flux = self._compute_dispersal_input(B)
            est_prob = self._get_est_prob(non_self_g)
            inv_rate = inv_flux * est_prob / BODY_MASS
            
            b_idx = self.buffer_idx
            self.buf_B[b_idx] = B.reshape(self.shape_2d)
            self.buf_W[b_idx] = (pclock < 0).reshape(self.shape_2d)
            self.buf_PC[b_idx] = pclock.reshape(self.shape_2d)
            self.buf_G[b_idx] = non_self_g.reshape(self.shape_2d)
            self.buf_INV[b_idx] = inv_rate.reshape(self.shape_2d)
            self.buf_EST[b_idx] = est_prob.reshape(self.shape_2d)
            
            self.time_points[self.global_idx] = t
            self.buffer_idx += 1; self.global_idx += 1
            if self.buffer_idx >= BUFFER_SIZE: self._flush_buffer()

        if self.buffer_idx > 0: self._flush_buffer()

        # --- DIAGNOSTICS ---
        y_final = solver.y
        sw_final = solver.sw.reshape(self.S, -1)
        logB_final = y_final[:total].reshape(self.S, -1)
        B_final = np.exp(np.minimum(logB_final, LOG_B_CAP))
        local_g_final, non_self_g_final = self._compute_local_growth(B_final)
        
        # 1. Standard Counts
        mask_S = sw_final
        mask_active = ~sw_final
        mask_P = mask_active & (non_self_g_final < 0)
        mask_D = mask_active & (non_self_g_final >= 0)
        
        count_S = int(np.sum(mask_S))
        count_P = int(np.sum(mask_P))
        count_D = int(np.sum(mask_D))
        count_Total = self.S * self.N_patches
        logger.info(f"[PSD2 STATS] Total: {count_Total} | S (Wait): {count_S} ({count_S/count_Total:.1%}) | P (Prob): {count_P} ({count_P/count_Total:.1%}) | D (Det): {count_D} ({count_D/count_Total:.1%})")

        # Masks for trait based on r in one patch:
        mask_type_0 = self.r_flat[:,0] > 0.9

        
        # 2. ASCII Histogram
        def draw_ascii_hist(data_tensor, bins=15, width=40, title=""):
            data = to_cpu(data_tensor).flatten()
            if len(data) == 0: return f"{title}\nNo data."
            counts, edges = _cpu_numpy.histogram(data, bins=bins)
            max_count = counts.max()
            if max_count == 0: max_count = 1
            out = [f"\n--- {title} (N={len(data)}) ---"]
            # for i in range(bins):
            #     bar_len = int((counts[i] / max_count) * width)
            #     bar = '#' * bar_len
            #     out.append(f"{edges[i]:8.4f} .. {edges[i+1]:8.4f} | {bar} ({counts[i]})")
            mean_val = _cpu_numpy.mean(data)
            std_val = _cpu_numpy.std(data)
            out.append(f"Stats: Mean={mean_val:.4f}, StdDev={std_val:.4f}")
            return "\n".join(out)

        mask_PS = mask_S | mask_P
        if np.any(mask_PS):
            logger.info(draw_ascii_hist(local_g_final[mask_PS], title="Local Growth Rates (P + S States)"))
            # logger.info(draw_ascii_hist(non_self_g_final[mask_PS], title="Non-Self Growth Rates (P + S States)"))
        else:
            logger.info("[PSD2 STATS] No P or S populations found for histogram.")

        occupancy = np.sum(mask_D,axis = 1)+0.0
        range_rarity_field = np.sum((mask_D/occupancy[:, None])[occupancy>0], axis = 0)
        range_rarity_field = range_rarity_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
        mean_range_rarity = \
            to_cpu(np.mean(range_rarity_field, axis = 1))
        csv_path = "range_rarity.csv"
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row_y", "mean_range_rarity"])
                for y in range(NUM_PATCHES_Y):
                    # y+1 for 1-based indexing as requested
                    writer.writerow([y+1, f"{mean_range_rarity[y]:.4f}"])
            
            logger.info(f"Written row-wise range_rarity to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to write {csv_path}: {e}")


        
        # <--- NEW: CSV Output by Row (High vs Low r) --->
        csv_path = "occupancy.csv"
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["speciesID", "occupancy", "type"])
                for y in range(len(occupancy)):
                    # y+1 for 1-based indexing as requested
                    writer.writerow([y+1, f"{occupancy[y]}", f"{mask_type_0[y]+1}"])
            
            logger.info(f"Written occupancy list to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to write {csv_path}: {e}")

        

        csv_path = "richness_by_type.csv"
        try:
            mask_type_0 = self.r_flat[:,0] > 0.9

            # Remove non-D biomass:
            # B_final.flatten()[mask_D.flatten()] = 0
            
            # Biomass per site of species of type 0 and 1:
            site_richness_type_0 = np.sum(mask_D[mask_type_0,:]+0.0, axis=0)
            site_richness_type_1 = np.sum(mask_D[~mask_type_0,:]+0.0, axis=0)
            
            # Make 2D biomass field:
            array_shape = (NUM_PATCHES_Y, NUM_PATCHES_X)
            richness_array_type_0 = \
                site_richness_type_0.reshape(array_shape)
            richness_array_type_1 = \
                site_richness_type_1.reshape(array_shape)

            # Mean over X (width) -> (Y,)
            # We must move to CPU before using standard Python CSV
            mean_y_0 = to_cpu(np.mean(richness_array_type_0, axis=1))
            mean_y_1 = to_cpu(np.mean(richness_array_type_1, axis=1))
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row_y", "mean_richess_0", "mean_richness_1"])
                for y in range(len(mean_y_0)):
                    # y+1 for 1-based indexing as requested
                    writer.writerow([y+1, f"{mean_y_0[y]:.4f}", f"{mean_y_1[y]:.4f}"])
            
            logger.info(f"Written row-wise richness stats to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to write {csv_path}: {e}")

        # Compute extinctions from destruction of half of type 2 environment.
        NY, NX = (NUM_PATCHES_Y, NUM_PATCHES_X)
        remaining_t1 = mask_D.reshape(self.S, NY, NX)[mask_type_0]
        remaining_t1[:, (NY//4):NY, (NX//2):NX] = 0
        remaining_t2 = mask_D.reshape(self.S, NY, NX)[~mask_type_0]
        remaining_t2[:, (NY//4):NY, (NX//2):NX] = 0
        predicted_extinctions_1 = \
            np.sum(np.sum(remaining_t1, axis=(1,2))==0)
        predicted_extinctions_2 = \
            np.sum(np.sum(remaining_t2, axis=(1,2))==0)
        print(f"Predicted_extinctions: {predicted_extinctions_1}/{np.sum(mask_type_0)}, {predicted_extinctions_2}/{np.sum(~mask_type_0)}")

        csv_path = "biomass_by_type.csv"
        try:
            # Reshape inputs to 2D grid: (S, Y, X)
            # sw_final is (S, P) -> (S, Y, X)
            # non_self_g_final is (S, P) -> (S, Y, X)
            # r_flat is (S, P) -> (S, Y, X)
            
            # Masks for trait based on r in one patch:
            mask_type_0 = self.r_flat[:,0] > 0.9

            # Remove non-D biomass:
            # B_final.flatten()[mask_D.flatten()] = 0
            
            # Biomass per site of species of type 0 and 1:
            site_biomass_type_0 = np.sum(B_final[mask_type_0,:], axis=0)
            site_biomass_type_1 = np.sum(B_final[~mask_type_0,:], axis=0)
            
            # Make 2D biomass field:
            array_shape = (NUM_PATCHES_Y, NUM_PATCHES_X)
            biomass_array_type_0 = \
                site_biomass_type_0.reshape(array_shape)
            biomass_array_type_1 = \
                site_biomass_type_1.reshape(array_shape)

            # Mean over X (width) -> (Y,)
            # We must move to CPU before using standard Python CSV
            mean_y_0 = to_cpu(np.mean(biomass_array_type_0, axis=1))
            mean_y_1 = to_cpu(np.mean(biomass_array_type_1, axis=1))
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row_y", "mean_B_0", "mean_B_1"])
                for y in range(len(mean_y_0)):
                    # y+1 for 1-based indexing as requested
                    writer.writerow([y+1, f"{mean_y_0[y]:.4f}", f"{mean_y_1[y]:.4f}"])
            
            logger.info(f"Written row-wise abundance stats to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to write {csv_path}: {e}")

        logger.info("PSD2 simulation completed.")
        return (self.time_points, self.trajectory, self.wait_trajectory, self.poisson_clock_traj, self.growth_rate_traj, self.invasion_rate_traj, self.establishment_prob_traj)
