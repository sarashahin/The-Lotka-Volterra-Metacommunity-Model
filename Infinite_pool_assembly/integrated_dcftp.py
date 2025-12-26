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

# Monkey-patch dispersal's dependencies to force CPU usage
dispersal.np = np  # Replace accelerator.np with standard numpy
dispersal.fft = FFTShimCPU  # Replace accelerator.fft with SciPy shim
dispersal.to_cpu = lambda x: x  # No-op since we are already on CPU

# Now we can safely use dispersal.compute_dispersal with numpy arrays

class IntegratedDCFTP:
    def __init__(self, species_type=1, csv_path='metacommunity_fields.csv'):
        """
        A DCFTP sampler that uses the 'Effective Metapopulation' theory.
        """
        self.L_y = NUM_PATCHES_Y
        self.L_x = NUM_PATCHES_X
        self.N = self.L_x * self.L_y
        self.ext_scaling = 1.0  # Default scaling factor
        
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
        
        # Precompute the kernel if needed (triggers the monkey-patched FFT)
        if config.DISPERSAL_KERNEL is not None:
            dispersal._precompute_custom_kernel((self.L_y, self.L_x))

        # Initialize effective extirpation rate
        self.set_extirpation_scaling(1.0)

        print(f"Initialized IntegratedDCFTP for Species Type {species_type}")
        print(f"  Grid: {self.L_y}x{self.L_x}")
        print(f"  Mean Abundance Range: {self.meanB_field.min():.3f} - {self.meanB_field.max():.3f}")
        print(f"  Base Extirpation Rate Range: {self.base_extRate_field.min():.3f} - {self.base_extRate_field.max():.3f}")

    def set_extirpation_scaling(self, factor):
        self.ext_scaling = factor
        self.survival_prob = np.exp(-self.base_extRate_field * self.ext_scaling)

    def get_coupled_rng(self, t, master_seed):
        hash_input = f"{master_seed}_{t}".encode('utf-8')
        step_seed = int(hashlib.sha256(hash_input).hexdigest(), 16) % (2**32)
        return np.random.default_rng(step_seed)

    def simulation_step(self, current_state, t, master_seed):
        """
        Advances the grid one step using pure CPU NumPy.
        """
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

    def generate_single_survivor(self, master_seed):
        # --- PHASE 1: FIND THE HORIZON (Backward) ---
        T_horizon = 1
        while True:
            state = np.ones(self.N, dtype=bool)
            for t in range(-T_horizon, 0):
                state = self.simulation_step(state, t, master_seed)
                if not np.any(state): break 
            if not np.any(state): break
            else:
                T_horizon *= 2
                if T_horizon > 10000: return None

        # --- PHASE 2: REJECTION SAMPLING (Forward) ---
        batch_size = 5
        attempts = 0
        arrival_seed = int(hashlib.sha256(f"arrival_{master_seed}".encode()).hexdigest(), 16) % (2**32)
        meta_rng = np.random.default_rng(arrival_seed)
        
        while True:
            attempts += 1
            arrival_times = meta_rng.integers(-T_horizon, 0, size=batch_size)
            survivors_in_batch = []
            
            for arrival_t in arrival_times:
                # Weighted start
                weights = self.pCol_field / self.pCol_field.sum()
                start_node = meta_rng.choice(self.N, p=weights)
                
                species_range = np.zeros(self.N, dtype=bool)
                species_range[start_node] = True
                
                for t in range(arrival_t, 0):
                    species_range = self.simulation_step(species_range, t, master_seed)
                    if not np.any(species_range): break 
                
                if np.any(species_range):
                    survivors_in_batch.append(species_range)
            
            if len(survivors_in_batch) > 0:
                return survivors_in_batch[meta_rng.integers(0, len(survivors_in_batch))]
            
            if attempts > 200: return np.zeros(self.N, dtype=bool)

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

def tune_combined_extirpation_scaling(models, target_p, max_iter=10, samples_per_iter=50):
    """
    Tunes scaling factor 's' to achieve a target Log-Series parameter 'p'.
    """
    print(f"\n--- Tuning Combined Extirpation Scaling (Target p: {target_p:.4f}) ---")
    
    history_s = []
    history_p = [] 
    
    current_s = 6.0
    s_min = current_s * 0.1
    s_max = current_s * 10
    
    for i in range(max_iter):
        for m in models: m.set_extirpation_scaling(current_s)
        
        all_occupancies = []
        n_exploded = 0
        
        seeds = [secrets.randbits(32) for _ in range(samples_per_iter)]
        for m in models:
            for s_seed in seeds:
                res = m.generate_single_survivor(s_seed)
                if res is None: n_exploded += 1
                elif np.any(res): all_occupancies.append(np.sum(res))
        
        total_samples = len(models) * samples_per_iter
        if n_exploded > total_samples * 0.5:
            print(f"Iter {i}: s={current_s:.4f} -> EXPLODED")
            history_s.append(current_s); history_p.append(1.0) # p -> 1 (critical)
            current_s = min(current_s * 1.5, s_max)
            continue
            
        if len(all_occupancies) == 0:
            print(f"Iter {i}: s={current_s:.4f} -> EXTINCT")
            history_s.append(current_s); history_p.append(0.0) # p -> 0
            current_s = max(current_s * 0.6, s_min)
            continue
            
        # Calculate Mean Occupancy of the batch
        mean_occ = np.mean(all_occupancies)
        
        # Calculate MLE p from mean
        obs_p = calculate_p_mle(mean_occ)
        
        history_s.append(current_s)
        history_p.append(obs_p)
        print(f"Iter {i}: s={current_s:.4f} -> Mean Occ={mean_occ:.2f}, p_mle={obs_p:.4f}")
                    
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
                
                # Prediction Error Calculation (Outcome Variable Error)
                x_arr = np.array(history_s); y_arr = np.array(history_p)
                n = len(x_arr)
                y_fit = slope * x_arr + intercept
                sse = np.sum((y_arr - y_fit)**2)
                sigma = np.sqrt(sse / (n - 2)) if n > 2 else 1.0
                x_mean = np.mean(x_arr); sxx = np.sum((x_arr - x_mean)**2)
                
                term_dist = (s_next - x_mean)**2 / sxx if sxx > 1e-9 else 0
                pred_error = sigma * np.sqrt(1 + 1/n + term_dist)
                rel_error = pred_error / (1-target_p) if target_p < 1 else float('inf')
            
            s_next = max(s_min, min(s_max, s_next))
            print(f"  Prediction: s_next={s_next:.4f} (RelErr={rel_error:.2%})")
            
            if rel_error < 0.1:
                print(f"  -> Converged!")
                return s_next
            current_s = s_next
        else:
            # Heuristic for early steps
            if obs_p > target_p:
                current_s *= 1.2
            else:
                current_s *= 0.8
                
    return current_s

def tile_simulation_grids(sample_grids, rows=None, cols=None, padding=1, pad_value=0.0):
    """
    Tiles a list of 2D grids into a single large image array.
    Used for creating a 'gallery' view of species ranges.
    
    Args:
        sample_grids: List of (Ny, Nx) numpy arrays.
        rows, cols: Layout geometry. If None, auto-calculated for ~16:9 aspect.
        padding: Pixels of padding between tiles.
        pad_value: Value for padding (0=empty/black, NaN=transparent/white depending on plot).
    
    Returns:
        Large 2D array containing the tiled grids.
    """
    N = len(sample_grids)
    if N == 0: return None
    
    Ny, Nx = sample_grids[0].shape
    
    if cols is None:
        # Heuristic for ~16:9 aspect ratio
        aspect = 16/9
        # solve: (cols * Nx) / (rows * Ny) ~ aspect  AND  rows * cols >= N
        # cols ~ rows * (Ny/Nx) * aspect
        # rows^2 * (Ny/Nx) * aspect >= N
        rows = int(np.sqrt(N / ((Ny/Nx) * aspect)))
        rows = max(1, rows)
        cols = int(np.ceil(N / rows))
    elif rows is None:
        rows = int(np.ceil(N / cols))
        
    # Create canvas
    H = rows * Ny + (rows - 1) * padding
    W = cols * Nx + (cols - 1) * padding
    
    # Initialize with pad_value
    # If pad_value is NaN, use float array. If 0, preserve type if possible.
    dtype = np.float32 if np.isnan(pad_value) else sample_grids[0].dtype
    canvas = np.full((H, W), pad_value, dtype=dtype)
    
    for i, grid in enumerate(sample_grids):
        if i >= rows * cols: break
        
        r, c = divmod(i, cols)
        
        y_start = r * (Ny + padding)
        x_start = c * (Nx + padding)
        
        canvas[y_start:y_start+Ny, x_start:x_start+Nx] = grid
        
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
        ax.plot(k_values, exp_counts, 'k--', linewidth=2, label=f'LogSer (p={p_hat:.2f})')
        
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
            ax.text(0.6, 0.6, f"Mean: {sample_mean:.1f}\nGoF p={p_val:.3f}", 
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
    except: pass
    
    ax.set_title(title); ax.set_yscale('log'); ax.legend(loc='upper right')
    ax.set_xlabel("Range Size"); ax.set_ylabel("Count")

if __name__ == "__main__":
    print("=== Integrated DCFTP Metacommunity Simulation (CPU-Only) ===")
    
    # 1. Initialize
    print("\nInitializing Models...")
    model_t1 = IntegratedDCFTP(species_type=1)
    model_t2 = IntegratedDCFTP(species_type=2)
    
    # 2. Tune
    target_p = calculate_p_mle(10.3) 
    optimal_s = tune_combined_extirpation_scaling([model_t1, model_t2], target_p)
    print(f"Optimal Combined Scaling: {optimal_s:.4f}")
    
    model_t1.set_extirpation_scaling(optimal_s)
    model_t2.set_extirpation_scaling(optimal_s)
    
    # 3. Generate
    num_samples = 400
    seeds = [secrets.randbits(32) for _ in range(num_samples)]
    
    print("\nGenerating Final Samples...")
    samples_t1 = []
    samples_t2 = []
    
    for i, s in enumerate(seeds):
        res1 = model_t1.generate_single_survivor(s)
        if res1 is not None and np.any(res1): samples_t1.append(res1)
        
        res2 = model_t2.generate_single_survivor(s)
        if res2 is not None and np.any(res2): samples_t2.append(res2)

    # 4. Plot Analysis
    print("\nGenerating Analysis Plots...")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    rows = np.arange(NUM_PATCHES_Y)
    ax[0,0].plot(rows, model_t1.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 1 MeanB')
    ax[0,0].plot(rows, model_t2.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], label='Type 2 MeanB')
    ax[0,0].set_title("Mean Abundance Profile"); ax[0,0].legend()
    
    if len(samples_t1) > 0:
        ax[0,1].imshow(samples_t1[0].reshape(NUM_PATCHES_Y, NUM_PATCHES_X), cmap='Blues', interpolation='nearest')
        ax[0,1].set_title(f"Example Range (Type 1)\nScaling: {optimal_s:.2f}")
    
    analyze_and_plot(ax[1,0], samples_t1, "Type 1 Occupancy", "skyblue")
    analyze_and_plot(ax[1,1], samples_t2, "Type 2 Occupancy", "salmon")
    
    plt.tight_layout(); plt.show()
    
    # 5. Movie-Style Tiled Layout
    print("\nGenerating Tiled Range Visualization...")
    # Combine and reshape samples
    all_grids = []
    # Mix samples to show diversity or just show one type? Let's mix.
    # Take up to 100 samples total to fit nicely on screen
    mix_limit = 100
    
    for i in range(mix_limit):
        if i < len(samples_t1): 
            all_grids.append(samples_t1[i].reshape(NUM_PATCHES_Y, NUM_PATCHES_X))
        if i < len(samples_t2):
            all_grids.append(samples_t2[i].reshape(NUM_PATCHES_Y, NUM_PATCHES_X))
            
    if all_grids:
        tiled_img = tile_simulation_grids(all_grids, padding=1, pad_value=np.nan)
        
        plt.figure(figsize=(16, 9))
        # Use a colormap where NaN (padding) is white/transparent and occupied is colored
        # 'viridis' or 'plasma' are good. Using 'Greens' for a biological feel.
        current_cmap = cm.get_cmap('viridis').copy()
        current_cmap.set_bad(color='white')
        
        plt.imshow(tiled_img, cmap=current_cmap, interpolation='nearest')
        plt.axis('off')
        plt.title(f"Sampled Species Ranges (Combined Types, n={len(all_grids)})")
        plt.tight_layout()
        plt.show()
        
    print("Done.")
