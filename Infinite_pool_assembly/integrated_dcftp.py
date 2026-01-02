############################################
# integrated_dcftp.py
############################################
import numpy as np
import pandas as pd
import csv
import statsmodels.api as sm   # for lineaer regression
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import logser, chi2, t
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

class IntegratedDCFTP:
    def __init__(self, species_type=1, csv_path='metacommunity_fields.csv', dt=0.1):
        self.L_y = NUM_PATCHES_Y
        self.L_x = NUM_PATCHES_X
        self.N = self.L_x * self.L_y
        self.ext_scaling = 1.0
        self.dt = dt
        
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

        # # FOR DEBUGGING:
        # test_field = np.ones_like(self.pCol_field)
        # for l in range(2):
        #     test_total = np.sum(test_field)
        #     print(test_total)
        #     test_abundance = test_field * self.meanB_field
        #     print(test_abundance[0])
        #     test_abundance_2d = test_abundance.reshape(1, self.L_y, self.L_x)
        #     print(test_abundance_2d[0,0,0])
        #     flux_field = dispersal.compute_dispersal(test_abundance_2d) / DISPERSAL_RATE 
        #     print(f"Incoming: {flux_field[0,0,0]}")
        #     invasion_field = flux_field.flatten() * self.pCol_field
        #     print(invasion_field[0])
        #     test_field += invasion_field - test_field * self.base_extRate_field
        #     print(test_field[0])
        #     print(f"change: {np.log(np.sum(test_field)/test_total)}")
        # endit

    def set_extirpation_scaling(self, factor):
        self.ext_scaling = factor
        self.survival_prob = np.exp(-self.base_extRate_field * (self.ext_scaling * self.dt))
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
        flux_field = flux_field / (DISPERSAL_RATE)
        flux_host = flux_field.flatten()
        
        # 3. Calculate Probability of Establishment
        colonization_rate = flux_host * self.pCol_field
        prob_colonization = 1.0 - np.exp(-colonization_rate * self.dt)
        
        # 4. Determine New Colonizations
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
                if t_horizon > 8*100000:
                    print(f"Explosion to t = {t_horizon}, Occ = {np.sum(state)}")
                    return None
        return t_horizon

def calculate_p_mle(mean_occupancy):
    """
    Calculates the Maximum Likelihood Estimate for the Log-Series parameter p
    given the sample mean occupancy (x_bar).
    
    Formula using Lambert W function:
    p_hat = 1 + 1 / ( x_bar * W_(-1)( -exp(-1/x_bar) / x_bar ) )
    
    Where W_(-1) is the lower branch of the Lambert W function.
    Reference: https://math.stackexchange.com/questions/3525734/maximum-likelihood-estimator-for-logarithmic-distribution
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
            
            # B. Horizon Check
            if arrival_t < -horizons[type_idx]:
                continue 
                
            # C. Sample Location (using chosen model's specific field)
            # Standard probability proportional to pCol field
            loc_weights = chosen_model.pCol_field / chosen_model.pCol_field.sum()
            start_node = meta_rng.choice(chosen_model.N, p=loc_weights)
            
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
            print(f"Horizon: {max_horizon}, {attempts}", end="                 \r")
            return survivors_in_batch[meta_rng.integers(0, len(survivors_in_batch))]
            
        if attempts > 20000:  # We need a heuristic for choosign this limit on attempts
            print("Attempts exhausted")
            return np.zeros(models[0].N, dtype=bool), -1

def tune_combined_extirpation_scaling(models, fractions, target_p, max_iter=100, samples_per_iter=30):
    print(f"\n--- Tuning Combined Extirpation Scaling (Target p: {target_p:.4f}) ---")
    history_s, history_p = [], []
    current_s = 1.05
    s_min = current_s * 0.1
    s_max = current_s * 10
    s_explodes = 0
    
    for i in range(max_iter):

        # remove old values that might bias the linear regression:
        if i % 10 == 9 and len(history_s) > 0:
            del history_s[0]
            del history_p[0]

        if current_s <= s_explodes:
            print(f"Iter {i}: s={current_s:.4f} -> EXPLODES")
            current_s = s_explodes
            if len(history_s) > 0:
                current_s = 0.5*(current_s + min(history_s))
            else:
                current_s = min(current_s * 1.5, s_max)
            continue
        
        for m in models: m.set_extirpation_scaling(current_s)
        
        occupancies = []
        n_exploded = 0
        seeds = [secrets.randbits(32) for _ in range(samples_per_iter)]
        
        for s_seed in seeds:
            print(f"{samples_per_iter - len(occupancies)}: ", end="")
            grid, _ = generate_mixed_survivor(models, fractions, s_seed)
            if grid is None: 
                n_exploded += 1
                if n_exploded > min(1, samples_per_iter * 0.5): break 
            elif np.any(grid): occupancies.append(np.sum(grid))
        
        if n_exploded > min(1, samples_per_iter * 0.5):
            print(f"Iter {i}: s={current_s:.4f} -> EXPLODED")
            if current_s > s_explodes:
                s_explodes = current_s
            if len(history_s) > 0:
                current_s = 0.5*(current_s + min(history_s))
            else:
                current_s = min(current_s * 1.5, s_max)
            continue
            
        if len(occupancies) == 0:
            print(f"Iter {i}: s={current_s:.4f} -> EXTINCT")
            current_s = max(current_s * 0.9, s_min)
            continue
            
        obs_p = calculate_p_mle(np.mean(occupancies))
        # Variance of log-series distribution as reference:
        theo_variance = -(obs_p**2 + obs_p * np.log(1-obs_p))/ \
            ( (1-obs_p) * np.log(1-obs_p) )**2
        history_s.append(current_s); history_p.append(obs_p)
        print(f"Iter {i}: s={current_s:.4f} -> Mean Occ={np.mean(occupancies):.2f}, var_Ex={np.var(occupancies)/theo_variance:.2f}, p_mle={obs_p:.4f}")
        
        if len(history_s) >= 3:
            # Linear Regression of p vs s
            # 1. Create data frame
            data = pd.DataFrame({
                's': history_s,
                'p': history_p
                })

            # 2. Add a constant (intercept) to the independent variable
            # statsmodels requires you to explicitly add the intercept column
            X = sm.add_constant(data['s'])
            y = data['p']

            # 3. Fit the model
            lin_model = sm.OLS(y,X).fit()

            # 3. Access specific values by column name
            intercept = lin_model.params['const']
            slope = lin_model.params['s']
        
            if abs(slope) < 1e-6:
                s_next = current_s * (0.8 if obs_p < target_p else 1.2)
                rel_error = float('inf')
            else:
                # Predict s_next where p = target_p
                # p = slope * s + intercept => s = (p - intercept) / slope
                s_next = (target_p - intercept) / slope

                # 4. Define the point you want to predict (e.g., x = 5.5)
                # Don't forget to add the constant (1.0) here as well!
                point_to_predict = pd.DataFrame({'const': [1.0], 's': [s_next]})
                prediction = lin_model.get_prediction(point_to_predict)
                pred_error_p = prediction.se_mean[0]
                
                # Use relative error on s as convergence metric
                rel_error = pred_error_p / (1-target_p) 
            
            s_next = max(s_min, min(s_max, s_next))
            print(f"  Prediction: s_next={s_next:.4f} (RelErr={rel_error:.2%})")
            
            if rel_error < 0.05:
                print(f"  -> Converged!")
                return s_next
            current_s = s_next
        else:
            if obs_p > target_p:
                current_s *= 1.1
            else:
                current_s *= 0.9
                if current_s <= s_explodes:
                    current_s = s_explodes
                    if len(history_s) > 0:
                        current_s = 0.5*(current_s + min(history_s))
                    else:
                        current_s = min(current_s * 1.15, s_max)
                
                
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
            
            stats_txt = f"Mean: {sample_mean:.1f}\nGoF $\\chi^2$={chi2_stat:.2f}\ndf={df}, p={p_val:.3f}"
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
    
    target_p1 = calculate_p_mle(88.02532) 
    optimal_s1 = tune_combined_extirpation_scaling([model1], [1], target_p1)
    # optimal_s1 = 0.9745
    print(f"Optimal Scaling: {optimal_s1:.7f}")
    target_p2 = calculate_p_mle(151.1307) 
    optimal_s2 = tune_combined_extirpation_scaling([model2], [1], target_p2)
    # optimal_s2 = 1.0613
    print(f"Optimal Scaling: {optimal_s2:.7f}")
    
    model1.set_extirpation_scaling(optimal_s1)
    model2.set_extirpation_scaling(optimal_s2)
    
    # Generate Samples
    num_samples1 = 158
    seeds1 = [secrets.randbits(32) for _ in range(num_samples1)]
    num_samples2 = 283
    seeds2 = [secrets.randbits(32) for _ in range(num_samples2)]
    
    samples_t1, samples_t2 = [], []
    all_grids_for_movie = [] # Tuple (grid, type)

    print("\nGenerating Final Mixed Samples...")
    for i, s in zip(range(num_samples1), seeds1):
        grid1, _ = generate_mixed_survivor([model1], [1], s)
        type_idx = 0
        print(f"Type 1: {num_samples1-i}")
        if grid1 is not None and np.any(grid1):
            # Store for analysis
            if type_idx == 0: samples_t1.append(grid1)
            else: samples_t2.append(grid1)
            
            if len(all_grids_for_movie) < 50:
                # Store grid with Type encoded: 1 for Type 1, 2 for Type 2
                # We multiply the boolean grid by (type_idx + 1)
                colored_grid = grid1.astype(float) * (type_idx + 1)
                colored_grid[colored_grid == 0] = np.nan # Transparent background
                all_grids_for_movie.append(colored_grid.reshape(NUM_PATCHES_Y, NUM_PATCHES_X))
    for i, s in zip(range(num_samples2), seeds2):
        grid2, _ = generate_mixed_survivor([model2], [1], s)
        type_idx = 1
        print(f"Type 2: {num_samples2-i}")
        if grid2 is not None and np.any(grid2):
            # Store for analysis
            if type_idx == 0: samples_t1.append(grid2)
            else: samples_t2.append(grid2)
            
            if len(all_grids_for_movie) < 100:
                # Store grid with Type encoded: 1 for Type 1, 2 for Type 2
                # We multiply the boolean grid by (type_idx + 1)
                colored_grid = grid2.astype(float) * (type_idx + 1)
                colored_grid[colored_grid == 0] = np.nan # Transparent background
                all_grids_for_movie.append(colored_grid.reshape(NUM_PATCHES_Y, NUM_PATCHES_X))

    # Output summary stats files
    all_t1 = np.array(samples_t1).reshape((len(samples_t1), NUM_PATCHES_Y, NUM_PATCHES_X))
    t1_richness = np.mean(np.sum(all_t1+0.0, axis=0), axis=1)
    all_t2 = np.array(samples_t2).reshape((len(samples_t2), NUM_PATCHES_Y, NUM_PATCHES_X))
    t2_richness = np.mean(np.sum(all_t2+0.0, axis=0), axis=1)
    csv_path = "richness_by_type_dcftp.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_y", "mean_richness_1", "mean_richness_2"])
            for y in range(NUM_PATCHES_Y):
                writer.writerow([y+1, f"{t1_richness[y]:.4f}", f"{t2_richness[y]:.4f}"])
                    
            print(f"Written row-wise richness stats to {csv_path}")
            
    except Exception as e:
        print(f"Failed to write {csv_path}: {e}")

    # Compute extinctions from destruction of half of type 2 environment.
    NY, NX = (NUM_PATCHES_Y, NUM_PATCHES_X)
    remaining_t1 = all_t1
    remaining_t1[:, (NY//4):NY, (NX//2):NX] = 0
    remaining_t2 = all_t2
    remaining_t2[:, (NY//4):NY, (NX//2):NX] = 0
    predicted_extinctions_1 = \
        np.sum(np.sum(remaining_t1, axis=(1,2))==0)
    predicted_extinctions_2 = \
        np.sum(np.sum(remaining_t2, axis=(1,2))==0)
    print(f"Predicted_extinctions: {predicted_extinctions_1}/{len(samples_t1)}, {predicted_extinctions_2}/{len(samples_t2)}")

    # Compute mean range rarity by row
    mask_D = np.concatenate((np.array(samples_t1), np.array(samples_t2)), axis = 0)
    occupancy = np.sum(mask_D,axis = 1)+0.0
    range_rarity_field = np.sum((mask_D/occupancy[:, None])[occupancy>0], axis = 0)
    range_rarity_field = range_rarity_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
    mean_range_rarity = \
        np.mean(range_rarity_field, axis = 1)
    csv_path = "range_rarity_DCFTP.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_y", "mean_range_rarity"])
            for y in range(NUM_PATCHES_Y):
                # y+1 for 1-based indexing as requested
                writer.writerow([y+1, f"{mean_range_rarity[y]:.4f}"])

        print(f"Written row-wise range_rarity to {csv_path}")

    except Exception as e:
        print(f"Failed to write {csv_path}: {e}")

        
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
        tiled = tile_simulation_grids(all_grids_for_movie, padding=1, pad_value=0)
        plt.figure(figsize=(16, 9))
        # Custom cmap: 1=Blue, 2=Red. 
        # We can use a discrete colormap. 
        # Matplotlib's 'coolwarm' maps low to blue, high to red.
        # nan is white.
        cmap = cm.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        
        plt.imshow(tiled+1, cmap=cmap, interpolation='nearest', vmin=0.5, vmax=2.5)
        plt.axis('off')
        plt.title(f"Mixed Species Ranges (Blue=Type 1, Red=Type 2)")
        plt.tight_layout()
        plt.show()
    
    print("Done.")
