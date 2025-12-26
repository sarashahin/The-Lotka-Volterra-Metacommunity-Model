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
# We bypass the accelerator to avoid GPU-CPU transfers in the tight DCFTP loop.
# This ensures all dispersal math happens in pure NumPy.

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

# Import config (pure python constants)
import config
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, DISPERSAL_RATE

# Import dispersal but OVERRIDE its dependencies before use
import dispersal
dispersal.np = np
dispersal.fft = FFTShimCPU
dispersal.to_cpu = lambda x: x

# Monkey-patch dispersal's dependencies to force CPU usage
dispersal.np = np  # Replace accelerator.np with standard numpy
dispersal.fft = FFTShimCPU  # Replace accelerator.fft with SciPy shim
dispersal.to_cpu = lambda x: x  # No-op since we are already on CPU

# Now we can safely use dispersal.compute_dispersal with numpy arrays

class IntegratedDCFTP:
    def __init__(self, species_type=1, csv_path='metacommunity_fields.csv'):
        self.L_y = NUM_PATCHES_Y
        self.L_x = NUM_PATCHES_X
        self.N = self.L_x * self.L_y
        self.ext_scaling = 1.0
        
        # 1. Load Effective Parameters from CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Extract columns based on species type
        suffix = str(species_type)
        col_pCol = f"pColonisation{suffix}"
        col_meanB = f"meanAbundance{suffix}"
        col_ext = f"extirpationRate{suffix}"
        
        if col_pCol not in df.columns:
            raise ValueError(f"Column {col_pCol} not found in CSV.")

        # 2. Map 1D (Row) parameters to 2D Grid
        raw_pCol = df[col_pCol].values
        raw_meanB = df[col_meanB].values
        raw_ext = df[col_ext].values
        
        # Broadcast to (Y, X) then flatten to (N,) for simulation logic
        self.pCol_field = np.repeat(raw_pCol[:, np.newaxis], self.L_x, axis=1).flatten()
        self.meanB_field = np.repeat(raw_meanB[:, np.newaxis], self.L_x, axis=1).flatten()
        self.base_extRate_field = np.repeat(raw_ext[:, np.newaxis], self.L_x, axis=1).flatten()
        
        # Pre-calculate spatial mean of pCol for weighting
        self.mean_pCol = np.mean(self.pCol_field)
        
        # Precompute the kernel if needed (triggers the monkey-patched FFT)
        if config.DISPERSAL_KERNEL is not None:
            dispersal._precompute_custom_kernel((self.L_y, self.L_x))

        # Initialize effective extirpation rate
        self.set_extirpation_scaling(1.0)
        self.horizon_cache = None

    def set_extirpation_scaling(self, factor):
        self.ext_scaling = factor
        self.survival_prob = np.exp(-self.base_extRate_field * self.ext_scaling)
        self.horizon_cache = None # Invalidate horizon cache

    def get_coupled_rng(self, t, master_seed):
        hash_input = f"{master_seed}_{t}".encode('utf-8')
        step_seed = int(hashlib.sha256(hash_input).hexdigest(), 16) % (2**32)
        return np.random.default_rng(step_seed)

    def simulation_step(self, current_state, t, master_seed):
        rng = self.get_coupled_rng(t, master_seed)
        rand_ext = rng.random(self.N)
        rand_col = rng.random(self.N)
        
        # --- A. Extinction ---
        survivors = current_state & (rand_ext < self.survival_prob)
        
        # --- B. Colonization ---
        # 1. Construct Effective Biomass Field
        eff_biomass_flat = current_state.astype(float) * self.meanB_field
        eff_biomass_2d = eff_biomass_flat.reshape(1, self.L_y, self.L_x)
        
        # 2. Compute Dispersal Flux (Pure CPU via monkey-patch)
        flux_field = dispersal.compute_dispersal(eff_biomass_2d)
        
        # Normalize flux (if needed by your specific scaling logic)
        flux_field = flux_field / (DISPERSAL_RATE / BODY_MASS)
        flux_host = flux_field.flatten()
        
        # 3. Calculate Probability of Establishment
        colonization_rate = (flux_host / BODY_MASS) * self.pCol_field
        prob_colonization = 1.0 - np.exp(-colonization_rate)
        
        # 4. Determine New Colonizations
        newly_colonized = (~current_state) & (rand_col < prob_colonization)
        
        return survivors | newly_colonized

    def find_horizon(self, master_seed):
        """Finds the coupling horizon T where a full grid goes extinct using the SPECIFIC seed for this species type."""
        t_horizon = 100
        while True:
            state = np.ones(self.N, dtype=bool)
            for t in range(-t_horizon, 0):
                state = self.simulation_step(state, t, master_seed)
                if not np.any(state): break 
            if not np.any(state): break
            else:
                t_horizon *= 2
                if t_horizon > 10000: return None
        return t_horizon

def calculate_p_mle(mean_occupancy):
    """
    Calculates the Maximum Likelihood Estimate for the Log-Series parameter p
    given the sample mean occupancy (x_bar).
    
    Formula using Lambert W function:
    p_hat = 1 + 1 / ( x_bar * W_(-1)( -exp(-1/x_bar) / x_bar ) )
    
    Where W_(-1) is the lower branch of the Lambert W function.
    Reference: https://math.stackexchange.com/a/3525752
    """
    if mean_occupancy <= 1.0:
        return 1e-9 # Limit p -> 0 as mean -> 1
    
    # Argument for Lambert W
    # z = -exp(-1/x_bar) / x_bar
    x_bar = mean_occupancy
    arg = -np.exp(-1.0 / x_bar) / x_bar
    
    # Evaluate W_(-1) branch
    # Note: scipy.special.lambertw returns complex by default if branch is ambiguous, 
    # but for real argument in (-1/e, 0) and branch -1, it returns real.
    w_val = lambertw(arg, k=-1)
    
    # We take the real part (it should be real theoretically)
    w_real = np.real(w_val)
    
    p_hat = 1.0 + 1.0 / (x_bar * w_real)
    
    # Clamp to safe range (0, 1)
    return max(1e-9, min(1.0 - 1e-9, p_hat))

def generate_mixed_survivor(models, fractions, master_seed):
    """
    Generates a single survivor from a mix of species types.
    Uses simplified Type-First sampling logic.
    """
    # 0. Generate Independent Seeds
    type_seeds = []
    for i in range(len(models)):
        s = int(hashlib.sha256(f"type_{i}_{master_seed}".encode()).hexdigest(), 16) % (2**32)
        type_seeds.append(s)

    # 1. Determine Horizons
    horizons = []
    for i, m in enumerate(models):
        h = m.find_horizon(type_seeds[i])
        if h is None: return None, None
        horizons.append(h)
    
    max_horizon = max(horizons)
    
    # 2. Calculate Global Type Weights
    # Weight_k = f_k * <P_col_k>
    global_weights = []
    for m, f in zip(models, fractions):
        global_weights.append(f * m.mean_pCol)
    
    global_weights = np.array(global_weights)
    total_weight = global_weights.sum()
    if total_weight == 0: return np.zeros(models[0].N, dtype=bool), -1
    
    type_probs = global_weights / total_weight
    
    # 3. Rejection Sampling
    arrival_seed = int(hashlib.sha256(f"arrival_{master_seed}".encode()).hexdigest(), 16) % (2**32)
    meta_rng = np.random.default_rng(arrival_seed)
    
    batch_size = 5
    attempts = 0
    
    while True:
        attempts += 1
        arrival_times = meta_rng.integers(-max_horizon, 0, size=batch_size)
        survivors_in_batch = []
        
        for arrival_t in arrival_times:
            # A. Sample Type First (using global weights)
            type_idx = meta_rng.choice(len(models), p=type_probs)
            chosen_model = models[type_idx]
            
            # B. Sample Location (using chosen model's specific field)
            # Standard probability proportional to pCol field
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
            
        if attempts > 200:  # We need a heuristic for choosign this limit on attempts
            return np.zeros(models[0].N, dtype=bool), -1

def tune_combined_extirpation_scaling(models, fractions, target_p, max_iter=100, samples_per_iter=30):
    print(f"\n--- Tuning Combined Extirpation Scaling (Target p: {target_p:.4f}) ---")
    history_s, history_p = [], []
    current_s = 6.0
    s_min = current_s * 0.1
    s_max = current_s * 10
    
    for i in range(max_iter):
        for m in models: m.set_extirpation_scaling(current_s)
        
        occupancies = []
        n_exploded = 0
        seeds = [secrets.randbits(32) for _ in range(samples_per_iter)]
        
        for s_seed in seeds:
            grid, _ = generate_mixed_survivor(models, fractions, s_seed)
            if grid is None:
                n_exploded += 1
                if n_exploded > 10: break
            elif np.any(grid): occupancies.append(np.sum(grid))
        
        if n_exploded > min(10,samples_per_iter * 0.5):
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
            # Linear Regression of p vs s
            slope, intercept, _, _, _ = linregress(history_s, history_p)
            
            if abs(slope) < 1e-6:
                s_next = current_s * (0.8 if obs_p < target_p else 1.2)
                rel_error = float('inf')
            else:
                # Predict s_next where p = target_p
                # p = slope * s + intercept => s = (p - intercept) / slope
                s_next = (target_p - intercept) / slope
                
                # --- PREDICTION ERROR CALCULATION ---
                # Calculate standard error of the inverse prediction s_next
                x_arr = np.array(history_s); y_arr = np.array(history_p)
                n = len(x_arr)
                y_fit = slope * x_arr + intercept
                sse = np.sum((y_arr - y_fit)**2)
                sigma = np.sqrt(sse / (n - 2)) if n > 2 else 1.0
                
                x_mean = np.mean(x_arr)
                sxx = np.sum((x_arr - x_mean)**2)
                
                # Formula for SE of inverse prediction x0 from y0:
                # SE(x0) = (sigma / |slope|) * sqrt(1 + 1/n + (y0 - y_mean)^2 / (slope^2 * Sxx))
                # Here x0 = s_next, y0 = target_p
                # Note: (y0 - y_mean) / slope approx (x0 - x_mean)
                
                if sxx > 1e-9:
                    term_dist = (target_p - np.mean(y_arr))**2 / (slope**2 * sxx)
                    pred_error_s = (sigma / abs(slope)) * np.sqrt(1 + 1/n + term_dist)
                else:
                    pred_error_s = float('inf')
                
                # Use relative error on s as convergence metric
                rel_error = pred_error_s / s_next if s_next > 0 else float('inf')
            
            s_next = max(s_min, min(s_max, s_next))
            print(f"  Prediction: s_next={s_next:.4f} (RelErr={rel_error:.2%})")
            
            if rel_error < 0.01:
                print(f"  -> Converged!")
                return s_next
            current_s = s_next
        else:
            current_s *= (0.8 if obs_p > target_p else 1.2)
                
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
    if len(range_sizes) == 0: ax.text(0.5,0.5,"No Data"); return

    bins = np.arange(0.5, max(range_sizes) + 1.5, 1)
    ax.hist(range_sizes, bins=bins, color=color, alpha=0.6, label='Observed')
    
    try:
        p_hat = calculate_p_mle(np.mean(range_sizes))
        k = np.arange(1, max(range_sizes)+1)
        ax.plot(k, logser.pmf(k, p_hat)*len(range_sizes), 'k--', label=f'p={p_hat:.2f}')
    except: pass
    ax.set_title(title); ax.set_yscale('log'); ax.legend()

if __name__ == "__main__":
    print("=== Integrated DCFTP (CPU-Only) ===")
    model1 = IntegratedDCFTP(species_type=1)
    model2 = IntegratedDCFTP(species_type=2)
    models = [model1, model2]
    fractions = [0.5, 0.5] # 50/50 input rain
    
    # Tuning
    target_p = calculate_p_mle(5) # for faster testing # calculate_p_mle(10.3) 
    optimal_s = tune_combined_extirpation_scaling(models, fractions, target_p)
    print(f"Optimal Scaling: {optimal_s:.4f}")
    for m in models: m.set_extirpation_scaling(optimal_s)
    
    # Generate Samples
    num_samples = 400
    seeds = [secrets.randbits(32) for _ in range(num_samples)]
    
    samples_t1, samples_t2 = [], []
    all_grids_for_movie = [] # Tuple (grid, type)
    
    print("\nGenerating Final Mixed Samples...")
    for s in seeds:
        grid, type_idx = generate_mixed_survivor(models, fractions, s)
        if grid is not None and np.any(grid):
            # Store for analysis
            if type_idx == 0: samples_t1.append(grid)
            else: samples_t2.append(grid)
            
            # Store for movie (only store first 100 to save memory/time)
            if len(all_grids_for_movie) < 100:
                # Store grid with Type encoded: 1 for Type 1, 2 for Type 2
                # We multiply the boolean grid by (type_idx + 1)
                colored_grid = grid.astype(float) * (type_idx + 1)
                colored_grid[colored_grid == 0] = np.nan # Transparent background
                all_grids_for_movie.append(colored_grid.reshape(NUM_PATCHES_Y, NUM_PATCHES_X))

    # Analysis Plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    rows = np.arange(NUM_PATCHES_Y)
    ax[0,0].plot(rows, model1.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 1')
    ax[0,0].plot(rows, model2.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 2')
    ax[0,0].legend(); ax[0,0].set_title("Mean Biomass")
    
    # Pie chart of composition
    ax[0,1].pie([len(samples_t1), len(samples_t2)], labels=['Type 1', 'Type 2'], autopct='%1.1f%%')
    ax[0,1].set_title("Steady State Composition")
    
    analyze_and_plot(ax[1,0], samples_t1, "Type 1 Occupancy", "skyblue")
    analyze_and_plot(ax[1,1], samples_t2, "Type 2 Occupancy", "salmon")
    plt.tight_layout(); plt.show()
    
    # Movie Tiled Plot
    if all_grids_for_movie:
        tiled = tile_simulation_grids(all_grids_for_movie, padding=1, pad_value=np.nan)
        plt.figure(figsize=(16, 9))
        # Custom cmap: 1=Blue, 2=Red. 
        # We can use a discrete colormap. 
        # Matplotlib's 'coolwarm' maps low to blue, high to red.
        # nan is white.
        cmap = cm.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        
        plt.imshow(tiled, cmap=cmap, interpolation='nearest', vmin=0.5, vmax=2.5)
        plt.axis('off')
        plt.title(f"Mixed Species Ranges (Blue=Type 1, Red=Type 2)")
        plt.tight_layout()
        plt.show()
    
    print("Done.")
