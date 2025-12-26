############################################
# integrated_dcftp.py
############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import logser, chi2, linregress, t
from scipy.optimize import brentq
from scipy.special import lambertw
import hashlib
import secrets
import sys
import os

# --- 1. FORCE CPU MODE & MONKEY PATCHING ---
import scipy.fft

class FFTShimCPU:
    """Drop-in replacement for accelerator.fft using SciPy (CPU)."""
    @staticmethod
    def rfft2(x, s=None, **kwargs): return scipy.fft.rfft2(x, s=s, workers=-1)
    @staticmethod
    def irfft2(x, s=None, **kwargs): return scipy.fft.irfft2(x, s=s, workers=-1)
    @staticmethod
    def fft2(x, s=None, **kwargs): return scipy.fft.fft2(x, s=s, workers=-1)
    @staticmethod
    def ifft2(x, s=None, **kwargs): return scipy.fft.ifft2(x, s=s, workers=-1)

import config
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, DISPERSAL_RATE

import dispersal
dispersal.np = np
dispersal.fft = FFTShimCPU
dispersal.to_cpu = lambda x: x

class IntegratedDCFTP:
    def __init__(self, species_type=1, csv_path='metacommunity_fields.csv'):
        self.L_y = NUM_PATCHES_Y
        self.L_x = NUM_PATCHES_X
        self.N = self.L_x * self.L_y
        self.ext_scaling = 1.0
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}")
            
        df = pd.read_csv(csv_path)
        suffix = str(species_type)
        col_pCol = f"pColonisation{suffix}"
        col_meanB = f"meanAbundance{suffix}"
        col_ext = f"extirpationRate{suffix}"
        
        if col_pCol not in df.columns:
            raise ValueError(f"Column {col_pCol} not found in CSV.")

        raw_pCol = df[col_pCol].values
        raw_meanB = df[col_meanB].values
        raw_ext = df[col_ext].values
        
        self.pCol_field = np.repeat(raw_pCol[:, np.newaxis], self.L_x, axis=1).flatten()
        self.meanB_field = np.repeat(raw_meanB[:, np.newaxis], self.L_x, axis=1).flatten()
        self.base_extRate_field = np.repeat(raw_ext[:, np.newaxis], self.L_x, axis=1).flatten()
        self.mean_pCol = np.mean(self.pCol_field)
        
        if config.DISPERSAL_KERNEL is not None:
            dispersal._precompute_custom_kernel((self.L_y, self.L_x))

        self.set_extirpation_scaling(1.0)
        self.horizon_cache = None

    def set_extirpation_scaling(self, factor):
        self.ext_scaling = factor
        self.survival_prob = np.exp(-self.base_extRate_field * self.ext_scaling)
        self.horizon_cache = None 

    def get_coupled_rng(self, t, master_seed):
        hash_input = f"{master_seed}_{t}".encode('utf-8')
        step_seed = int(hashlib.sha256(hash_input).hexdigest(), 16) % (2**32)
        return np.random.default_rng(step_seed)

    def simulation_step(self, current_state, t, master_seed):
        rng = self.get_coupled_rng(t, master_seed)
        rand_ext = rng.random(self.N)
        rand_col = rng.random(self.N)
        
        survivors = current_state & (rand_ext < self.survival_prob)
        
        eff_biomass_flat = current_state.astype(float) * self.meanB_field
        eff_biomass_2d = eff_biomass_flat.reshape(1, self.L_y, self.L_x)
        flux_field = dispersal.compute_dispersal(eff_biomass_2d)
        flux_field = flux_field / (DISPERSAL_RATE / BODY_MASS)
        flux_host = flux_field.flatten()
        
        colonization_rate = (flux_host / BODY_MASS) * self.pCol_field
        prob_colonization = 1.0 - np.exp(-colonization_rate)
        
        newly_colonized = (~current_state) & (rand_col < prob_colonization)
        
        return survivors | newly_colonized

    def find_horizon(self, master_seed):
        """Finds the coupling horizon T where a full grid goes extinct using the SPECIFIC seed."""
        t_horizon = 100
        while True:
            state = np.ones(self.N, dtype=bool)
            for t in range(-t_horizon, 0):
                state = self.simulation_step(state, t, master_seed)
                if not np.any(state): break 
            if not np.any(state): break
            else:
                t_horizon *= 2
                if t_horizon > 100000: return None
        return t_horizon

def calculate_p_mle(mean_occupancy):
    if mean_occupancy <= 1.0: return 1e-9
    x_bar = mean_occupancy
    arg = -np.exp(-1.0 / x_bar) / x_bar
    w_val = lambertw(arg, k=-1)
    w_real = np.real(w_val)
    p_hat = 1.0 + 1.0 / (x_bar * w_real)
    return max(1e-9, min(1.0 - 1e-9, p_hat))

def generate_mixed_survivor(models, fractions, master_seed):
    """
    Generates a single survivor from a mix of species types.
    """
    type_seeds = []
    for i in range(len(models)):
        s = int(hashlib.sha256(f"type_{i}_{master_seed}".encode()).hexdigest(), 16) % (2**32)
        type_seeds.append(s)

    horizons = []
    for i, m in enumerate(models):
        h = m.find_horizon(type_seeds[i])
        if h is None: return None, None
        horizons.append(h)
    
    max_horizon = max(horizons)
    
    global_weights = []
    for m, f in zip(models, fractions):
        global_weights.append(f * m.mean_pCol)
    
    global_weights = np.array(global_weights)
    total_weight = global_weights.sum()
    if total_weight == 0: return np.zeros(models[0].N, dtype=bool), -1
    
    type_probs = global_weights / total_weight
    
    arrival_seed = int(hashlib.sha256(f"arrival_{master_seed}".encode()).hexdigest(), 16) % (2**32)
    meta_rng = np.random.default_rng(arrival_seed)
    
    batch_size = 5
    attempts = 0
    
    while True:
        attempts += 1
        arrival_times = meta_rng.integers(-max_horizon, 0, size=batch_size)
        survivors_in_batch = []
        
        for arrival_t in arrival_times:
            # A. Sample Type First
            type_idx = meta_rng.choice(len(models), p=type_probs)
            chosen_model = models[type_idx]
            
            # B. Sample Location
            loc_weights = chosen_model.pCol_field / chosen_model.pCol_field.sum()
            start_node = meta_rng.choice(chosen_model.N, p=loc_weights)
            
            # C. Horizon Check
            if arrival_t < -horizons[type_idx]:
                continue 
                
            # D. Simulate
            current_seed = type_seeds[type_idx]
            species_range = np.zeros(chosen_model.N, dtype=bool)
            species_range[start_node] = True
            
            for t in range(arrival_t, 0):
                species_range = chosen_model.simulation_step(species_range, t, current_seed)
                if not np.any(species_range): break 
            
            if np.any(species_range):
                survivors_in_batch.append((species_range, type_idx))
        
        if len(survivors_in_batch) > 0:
            return survivors_in_batch[meta_rng.integers(0, len(survivors_in_batch))]
            
        if attempts > 200:
            return np.zeros(models[0].N, dtype=bool), -1

def tune_combined_extirpation_scaling(models, fractions, target_p, max_iter=100, samples_per_iter=50):
    print(f"\n--- Tuning Combined Extirpation Scaling (Target p: {target_p:.4f}) ---")
    history_s, history_p = [], []
    current_s = 6.0
    s_min, s_max = 0.6, 60.0
    
    for i in range(max_iter):
        for m in models: m.set_extirpation_scaling(current_s)
        
        occupancies = []
        n_exploded = 0
        seeds = [secrets.randbits(32) for _ in range(samples_per_iter)]
        
        for s_seed in seeds:
            grid, _ = generate_mixed_survivor(models, fractions, s_seed)
            if grid is None: 
                n_exploded += 1
                if n_exploded > samples_per_iter * 0.5: break 
            elif np.any(grid): occupancies.append(np.sum(grid))
        
        if n_exploded > min(10, samples_per_iter * 0.5):
            print(f"Iter {i}: s={current_s:.4f} -> EXPLODED")
            history_s.append(current_s); history_p.append(1.0)
            current_s = min(current_s * 1.5, s_max)
            continue
            
        if len(occupancies) == 0:
            print(f"Iter {i}: s={current_s:.4f} -> EXTINCT")
            history_s.append(current_s); history_p.append(0.0)
            current_s = max(current_s * 0.6, s_min)
            continue
            
        obs_p = calculate_p_mle(np.mean(occupancies))
        history_s.append(current_s); history_p.append(obs_p)
        print(f"Iter {i}: s={current_s:.4f} -> Mean Occ={np.mean(occupancies):.2f}, p_mle={obs_p:.4f}")
        
        if len(history_s) >= 3:
            slope, intercept, _, _, _ = linregress(history_s, history_p)
            
            if abs(slope) < 1e-6:
                s_next = current_s * (0.8 if obs_p < target_p else 1.2)
                rel_error = float('inf')
            else:
                s_next = (target_p - intercept) / slope
                
                # --- PREDICTION ERROR CALCULATION (Restored) ---
                x_arr = np.array(history_s); y_arr = np.array(history_p)
                n = len(x_arr)
                y_fit = slope * x_arr + intercept
                sse = np.sum((y_arr - y_fit)**2)
                sigma = np.sqrt(sse / (n - 2)) if n > 2 else 1.0
                x_mean = np.mean(x_arr); sxx = np.sum((x_arr - x_mean)**2)
                
                if sxx > 1e-9:
                    # Inverse prediction error approximation
                    # Predicting x (s) from y (p)
                    term_dist = (target_p - np.mean(y_arr))**2 / (slope**2 * sxx)
                    pred_error_s = (sigma / abs(slope)) * np.sqrt(1 + 1/n + term_dist)
                else:
                    pred_error_s = float('inf')
                
                rel_error = pred_error_s / s_next if s_next > 0 else float('inf')
            
            s_next = max(s_min, min(s_max, s_next))
            print(f"  Prediction: s_next={s_next:.4f} (RelErr={rel_error:.2%})")
            
            if rel_error < 0.05:
                print(f"  -> Converged!")
                return s_next
            current_s = s_next
        else:
            if obs_p > target_p:
                current_s *= 1.2
            else:
                current_s *= 0.8
                
    return current_s

def tile_simulation_grids(sample_grids, rows=None, cols=None, padding=1, pad_value=0.0):
    N = len(sample_grids)
    if N == 0: return None
    Ny, Nx = sample_grids[0].shape
    
    if cols is None:
        aspect = 16/9
        rows = max(1, int(np.sqrt(N / ((Ny/Nx) * aspect))))
        cols = int(np.ceil(N / rows))
    elif rows is None: rows = int(np.ceil(N / cols))
        
    H, W = rows * Ny + (rows - 1) * padding, cols * Nx + (cols - 1) * padding
    
    dtype = np.float32 if np.isnan(pad_value) else sample_grids[0].dtype
    canvas = np.full((H, W), pad_value, dtype=dtype)
    
    for i, grid in enumerate(sample_grids):
        if i >= rows * cols: break
        r, c = divmod(i, cols)
        y, x = r * (Ny + padding), c * (Nx + padding)
        canvas[y:y+Ny, x:x+Nx] = grid
        
    return canvas

def analyze_and_plot(ax, samples, title, color):
    range_sizes = [np.sum(r) for r in samples]
    range_sizes = np.array([r for r in range_sizes if r > 0])
    
    if len(range_sizes) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center')
        return

    max_val = max(range_sizes)
    bins = np.arange(0.5, max_val + 1.5, 1)
    
    ax.hist(range_sizes, bins=bins, color=color, alpha=0.6, edgecolor='black', label='Observed')
    
    try:
        sample_mean = np.mean(range_sizes)
        p_hat = calculate_p_mle(sample_mean)
        
        k_values = np.arange(1, max_val + 1)
        exp_counts = logser.pmf(k_values, p_hat) * len(range_sizes)
        ax.plot(k_values, exp_counts, 'r--', linewidth=2, label=f'LogSer Fit (p={p_hat:.2f})')
        
        obs_counts = np.bincount(range_sizes)[1:]
        if len(obs_counts) < len(exp_counts):
             obs_counts = np.append(obs_counts, np.zeros(len(exp_counts) - len(obs_counts)))
        
        bin_o, bin_e, curr_o, curr_e = [], [], 0, 0
        for o, e in zip(obs_counts, exp_counts):
            curr_o += o; curr_e += e
            if curr_e >= 5:
                bin_o.append(curr_o); bin_e.append(curr_e); curr_o=0; curr_e=0
        if curr_e > 0:
            if bin_e: bin_e[-1]+=curr_e; bin_o[-1]+=curr_o
            else: bin_e.append(curr_e); bin_o.append(curr_o)
            
        bin_o = np.array(bin_o); bin_e = np.array(bin_e)
        if len(bin_o) > 0:
            chi2_stat = np.sum((bin_o - bin_e)**2 / bin_e)
            df = max(1, len(bin_o) - 2)
            p_val = 1 - chi2.cdf(chi2_stat, df)
            
            stats_txt = f"Mean: {sample_mean:.1f}\nGoF $\chi^2$={chi2_stat:.2f}\ndf={df}, p={p_val:.3f}"
            ax.text(0.5, 0.5, stats_txt, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=10)
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_xlabel("Range Size (Patches)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

if __name__ == "__main__":
    print("=== Integrated DCFTP (CPU-Only) ===")
    model1 = IntegratedDCFTP(species_type=1)
    model2 = IntegratedDCFTP(species_type=2)
    models = [model1, model2]
    fractions = [0.5, 0.5] 
    
    target_p = calculate_p_mle(5) 
    optimal_s = tune_combined_extirpation_scaling(models, fractions, target_p)
    print(f"Optimal Scaling: {optimal_s:.4f}")
    for m in models: m.set_extirpation_scaling(optimal_s)
    
    num_samples = 400
    seeds = [secrets.randbits(32) for _ in range(num_samples)]
    
    samples_t1, samples_t2 = [], []
    all_grids_for_movie = [] 
    
    print("\nGenerating Final Mixed Samples...")
    for s in seeds:
        grid, type_idx = generate_mixed_survivor(models, fractions, s)
        if grid is not None and np.any(grid):
            if type_idx == 0: samples_t1.append(grid)
            else: samples_t2.append(grid)
            
            if len(all_grids_for_movie) < 100:
                colored_grid = grid.astype(float) * (type_idx + 1)
                colored_grid[colored_grid == 0] = np.nan 
                all_grids_for_movie.append(colored_grid.reshape(NUM_PATCHES_Y, NUM_PATCHES_X))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    rows = np.arange(NUM_PATCHES_Y)
    ax[0,0].plot(rows, model1.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 1')
    ax[0,0].plot(rows, model2.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 2')
    ax[0,0].legend(); ax[0,0].set_title("Mean Biomass")
    
    ax[0,1].pie([len(samples_t1), len(samples_t2)], labels=['Type 1', 'Type 2'], autopct='%1.1f%%')
    ax[0,1].set_title("Steady State Composition")
    
    analyze_and_plot(ax[1,0], samples_t1, "Type 1 Occupancy", "skyblue")
    analyze_and_plot(ax[1,1], samples_t2, "Type 2 Occupancy", "salmon")
    plt.tight_layout(); plt.show()
    
    if all_grids_for_movie:
        tiled = tile_simulation_grids(all_grids_for_movie, padding=1, pad_value=np.nan)
        plt.figure(figsize=(16, 9))
        cmap = cm.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        plt.imshow(tiled, cmap=cmap, interpolation='nearest', vmin=0.5, vmax=2.5)
        plt.axis('off')
        plt.title(f"Mixed Species Ranges (Blue=Type 1, Red=Type 2)")
        plt.tight_layout()
        plt.show()
    
    print("Done.")
