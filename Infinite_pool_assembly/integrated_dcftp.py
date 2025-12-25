############################################
# integrated_dcftp.py
############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import logser, chi2, linregress, t
from scipy.optimize import brentq
import hashlib
import secrets
import sys
import os

# --- Import from Uploaded Metacommunity Codebase ---
# We use the accelerator shim and dispersal logic to ensure physics match exactly
import config
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, DISPERSAL_RATE
import dispersal
from accelerator import to_cpu, np as accel_np

class IntegratedDCFTP:
    def __init__(self, species_type=1, csv_path='metacommunity_fields.csv'):
        """
        A DCFTP sampler that uses the 'Effective Metapopulation' theory.
        
        Args:
            species_type (int): 1 or 2. Selects the column set from the CSV.
            csv_path (str): Path to the parameters CSV.
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
        # Assuming CSV has 20 rows corresponding to y=0..19
        suffix = str(species_type)
        col_pCol = f"pColonisation{suffix}"
        col_meanB = f"meanAbundance{suffix}"
        col_ext = f"extirpationRate{suffix}"
        
        if col_pCol not in df.columns:
            raise ValueError(f"Column {col_pCol} not found in CSV.")

        # 2. Map 1D (Row) parameters to 2D Grid
        # Data is (Y,), we broadcast to (Y, X)
        # We perform calculations on CPU numpy for CFTP stability, 
        # but convert to accelerator for dispersal step.
        
        raw_pCol = df[col_pCol].values
        raw_meanB = df[col_meanB].values
        raw_ext = df[col_ext].values
        
        # Broadcast to (Y, X) then flatten to (N,) for simulation logic
        self.pCol_field = np.repeat(raw_pCol[:, np.newaxis], self.L_x, axis=1).flatten()
        self.meanB_field = np.repeat(raw_meanB[:, np.newaxis], self.L_x, axis=1).flatten()
        self.base_extRate_field = np.repeat(raw_ext[:, np.newaxis], self.L_x, axis=1).flatten()
        
        # Initialize effective extirpation rate
        self.set_extirpation_scaling(1.0)

        print(f"Initialized IntegratedDCFTP for Species Type {species_type}")
        print(f"  Grid: {self.L_y}x{self.L_x}")
        print(f"  Mean Abundance Range: {self.meanB_field.min():.3f} - {self.meanB_field.max():.3f}")
        print(f"  Base Extirpation Rate Range: {self.base_extRate_field.min():.3f} - {self.base_extRate_field.max():.3f}")

    def set_extirpation_scaling(self, factor):
        """
        Scales the extirpation rates by a factor.
        Used to tune the system towards criticality.
        """
        self.ext_scaling = factor
        # Recalculate survival probabilities
        # survival_prob = exp( - rate * factor )
        self.survival_prob = np.exp(-self.base_extRate_field * self.ext_scaling)

    def get_coupled_rng(self, t, master_seed):
        """
        Deterministic RNG based on time step t and master seed.
        Ensures CFTP coupling works correctly.
        """
        hash_input = f"{master_seed}_{t}".encode('utf-8')
        step_seed = int(hashlib.sha256(hash_input).hexdigest(), 16) % (2**32)
        return np.random.default_rng(step_seed)

    def simulation_step(self, current_state, t, master_seed):
        """
        Advances the grid one step.
        """
        rng = self.get_coupled_rng(t, master_seed)
        
        # Generate random numbers for this step
        rand_ext = rng.random(self.N)
        rand_col = rng.random(self.N)
        
        # --- A. Extinction ---
        # Survive if random < survival_prob (which includes ext_scaling)
        survivors = current_state & (rand_ext < self.survival_prob)
        
        # --- B. Colonization ---
        # 1. Construct Effective Biomass Field
        eff_biomass_flat = current_state.astype(float) * self.meanB_field
        
        # 2. Compute Dispersal Flux (Using imported dispersal module)
        eff_biomass_2d = eff_biomass_flat.reshape(1, self.L_y, self.L_x)
        
        # Move to accelerator (GPU/CPU shim)
        # FIX: Ensure float32 for MPS compatibility
        b_device = accel_np.asarray(eff_biomass_2d.astype(np.float32))
        
        # Calculate Flux (includes DISPERSAL_RATE from config)
        flux_device = dispersal.compute_dispersal(b_device)/(DISPERSAL_RATE/BODY_MASS)
        
        # Move back to Host CPU for CFTP logic
        flux_host = to_cpu(flux_device).flatten()
        
        # 3. Calculate Probability of Establishment
        # Rate = Flux (biomass/time) / BODY_MASS (biomass/ind) * pColonisation (prob/ind)
        colonization_rate = (flux_host / BODY_MASS) * self.pCol_field
        
        # Probability of at least one successful colonization event in step dt=1
        prob_colonization = 1.0 - np.exp(-colonization_rate)
        
        # 4. Determine New Colonizations
        newly_colonized = (~current_state) & (rand_col < prob_colonization)
        
        return survivors | newly_colonized

    def generate_single_survivor(self, master_seed):
        """
        Runs the Dominated Coupling From The Past (DCFTP) algorithm
        to generate one perfect sample from the steady state distribution.
        Returns None if system appears supercritical (explodes).
        """
        # --- PHASE 1: FIND THE HORIZON (Backward) ---
        T_horizon = 1
        while True:
            # Max Chain: Start with full occupancy
            state = np.ones(self.N, dtype=bool)
            
            # Run forward from -T to 0
            for t in range(-T_horizon, 0):
                state = self.simulation_step(state, t, master_seed)
                if not np.any(state): break 
            
            if not np.any(state):
                # Max chain died out, horizon found
                break
            else:
                T_horizon *= 2
                if T_horizon > 10000: 
                    # System might be supercritical or T is too large
                    # Treat this as an "explosion"
                    return None

        # --- PHASE 2: REJECTION SAMPLING (Forward) ---
        batch_size = 5
        attempts = 0
        
        # Helper RNG for arrival times (uncoupled from grid dynamics)
        arrival_seed = int(hashlib.sha256(f"arrival_{master_seed}".encode()).hexdigest(), 16) % (2**32)
        meta_rng = np.random.default_rng(arrival_seed)
        
        while True:
            attempts += 1
            arrival_times = meta_rng.integers(-T_horizon, 0, size=batch_size)
            survivors_in_batch = []
            
            for arrival_t in arrival_times:
                # Initialize one random patch (weighted by pCol)
                weights = self.pCol_field / self.pCol_field.sum()
                start_node = meta_rng.choice(self.N, p=weights)
                
                species_range = np.zeros(self.N, dtype=bool)
                species_range[start_node] = True
                
                # Evolve using COUPLED master_seed
                for t in range(arrival_t, 0):
                    species_range = self.simulation_step(species_range, t, master_seed)
                    if not np.any(species_range): break 
                
                if np.any(species_range):
                    survivors_in_batch.append(species_range)
            
            if len(survivors_in_batch) > 0:
                # Return one survivor randomly
                return survivors_in_batch[meta_rng.integers(0, len(survivors_in_batch))]
            
            if attempts > 200: # 1000 attempts * 5 batch size
                # Likely empty world or very low survival
                return np.zeros(self.N, dtype=bool)

def tune_combined_extirpation_scaling(models, target_inv_mu, max_iter=100, samples_per_iter=4):
    """
    Search algorithm to find the single scaling factor for extirpation rates
    that yields the target mean inverse occupancy across both species types.
    
    Args:
        models: List of IntegratedDCFTP objects (e.g. [model_t1, model_t2])
        target_inv_mu: Target value for E[1/N]
    """
    print(f"\n--- Tuning Combined Extirpation Scaling (Target Mean 1/N: {target_inv_mu:.4f}) ---")
    
    history_s = []
    history_inv_mu = [] # Observed E[1/N]
    
    # Initial bounds / guesses
    current_s = 6.0
    
    # Safety bounds for scaling factor
    s_min = current_s * 0.1
    s_max = current_s * 10
    
    for i in range(max_iter):
        # Apply scaling to all models
        for m in models:
            m.set_extirpation_scaling(current_s)
        
        # Run batch of simulations for each model
        all_occupancies = []
        n_exploded = 0
        n_extinct = 0
        
        seeds = [secrets.randbits(32) for _ in range(samples_per_iter)]
        
        for m in models:
            for s_seed in seeds:
                res = m.generate_single_survivor(s_seed)
                if res is None:
                    n_exploded += 1
                elif np.any(res):
                    all_occupancies.append(np.sum(res))
                else:
                    n_extinct += 1 # Should theoretically not happen if generate_single_survivor works
        
        total_samples = len(models) * samples_per_iter
        
        if n_exploded > total_samples * 0.5:
            # Too supercritical
            print(f"Iter {i}: s={current_s:.4f} -> EXPLODED (System Supercritical)")
            history_s.append(current_s)
            history_inv_mu.append(0.0) # 1/inf -> 0
            current_s = min(current_s * 1.5, s_max)
            continue
            
        if len(all_occupancies) == 0:
            print(f"Iter {i}: s={current_s:.4f} -> EXTINCT (No survivors found)")
            history_s.append(current_s)
            history_inv_mu.append(1.0) # 1/1 = 1 (Max possible inv_mu)
            current_s = max(current_s * 0.6, s_min)
            continue
            
        # Calculate statistic: Mean Inverse Occupancy
        # E[1/N]
        vals = np.array(all_occupancies)
        inv_vals = 1.0 / vals
        obs_inv_mu = np.mean(inv_vals)
        
        history_s.append(current_s)
        history_inv_mu.append(obs_inv_mu)
        
        print(f"Iter {i}: s={current_s:.4f} -> Mean 1/N={obs_inv_mu:.4f}")
                    
        # Refine Estimate: Linear Regression on (s, inv_mu)
        # We expect 1/N to be roughly proportional to s (extirpation rate)
        # near criticality. 
        # inv_mu = slope * s + intercept
        
        if len(history_s) >= 3:
            # Linear Regression
            slope, intercept, r_val, _, stderr = linregress(history_s, history_inv_mu)
            
            if abs(slope) < 1e-6:
                # Flat line (bad fit), heuristic step
                if obs_inv_mu > target_inv_mu: # 1/N too high => N too low => ext too high
                    s_next = current_s * 0.8 # Decrease s
                else:
                    s_next = current_s * 1.2
                prediction_error = float('inf')
            else:
                # Predict s_next where inv_mu = target_inv_mu
                s_next = (target_inv_mu - intercept) / slope
                
                # --- PREDICTION ERROR CALCULATION ---
                # Calculate SE of the predicted value y_hat at x = s_next
                # SE_y_pred = sigma * sqrt(1 + 1/n + (s_next - s_bar)^2 / Sxx)
                
                x_arr = np.array(history_s)
                y_arr = np.array(history_inv_mu)
                n = len(x_arr)
                
                # Residual Standard Error (sigma)
                y_fit = slope * x_arr + intercept
                residuals = y_arr - y_fit
                sse = np.sum(residuals**2)
                sigma = np.sqrt(sse / (n - 2)) if n > 2 else 1.0
                
                # Sxx
                x_mean = np.mean(x_arr)
                sxx = np.sum((x_arr - x_mean)**2)
                
                if sxx > 1e-9:
                    term_dist = (s_next - x_mean)**2 / sxx
                    se_y_pred = sigma * np.sqrt(1 + 1/n + term_dist)
                else:
                    se_y_pred = float('inf')
                
                prediction_error = se_y_pred

            # Dampening / Bounds
            s_next = max(s_min, min(s_max, s_next))
            
            # Convergence Check
            # Error of the prediction (in units of 1/N) relative to the target
            rel_error = prediction_error / target_inv_mu
            
            print(f"  Prediction: s_next={s_next:.4f}, SE(pred)={prediction_error:.4f} (RelErr={rel_error:.2%})")
            
            if rel_error < 0.05: # 5% relative error on the outcome variable
                print(f"  -> Converged! (Relative Prediction Error: {rel_error:.2%})")
                return s_next
                
            current_s = s_next
        else:
            # Simple heuristic for early steps
            if obs_inv_mu > target_inv_mu: # N too small -> decrease s
                current_s *= 0.8
            else:
                current_s *= 1.2
                
    return current_s

def analyze_and_plot(ax, samples, title, color):
    """Fits Log-Series and plots histogram + GOF"""
    range_sizes = [np.sum(r) for r in samples]
    range_sizes = np.array(range_sizes)
    range_sizes = range_sizes[range_sizes > 0]
    
    if len(range_sizes) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center')
        return

    max_val = max(range_sizes)
    bins = np.arange(0.5, max_val + 1.5, 1)
    
    ax.hist(range_sizes, bins=bins, color=color, alpha=0.6, edgecolor='black', label='Observed')
    
    try:
        # Fit p using method of moments
        sample_mean = np.mean(range_sizes)
        def eq(p): 
            if p >= 1 or p <= 0: return 1e9
            return -p / ((1 - p) * np.log(1 - p)) - sample_mean
        
        p_hat = brentq(eq, 1e-9, 1.0 - 1e-9)
        
        # Expected counts
        k_values = np.arange(1, max_val + 1)
        exp_probs = logser.pmf(k_values, p_hat)
        exp_counts = exp_probs * len(range_sizes)
        
        ax.plot(k_values, exp_counts, 'k--', linewidth=2, label=f'LogSer (p={p_hat:.2f})')
        
        # Chi2 with Dynamic Binning
        obs_counts = np.bincount(range_sizes)[1:]
        if len(obs_counts) < len(exp_counts):
             obs_counts = np.append(obs_counts, np.zeros(len(exp_counts) - len(obs_counts)))
        
        bin_o, bin_e = [], []
        curr_o, curr_e = 0, 0
        for o, e in zip(obs_counts, exp_counts):
            curr_o += o; curr_e += e
            if curr_e >= 5:
                bin_o.append(curr_o); bin_e.append(curr_e)
                curr_o = 0; curr_e = 0
        if curr_e > 0:
            if bin_e: bin_e[-1]+=curr_e; bin_o[-1]+=curr_o
            else: bin_e.append(curr_e); bin_o.append(curr_o)
            
        bin_o = np.array(bin_o); bin_e = np.array(bin_e)
        if len(bin_o) > 0:
            chi2_stat = np.sum((bin_o - bin_e)**2 / bin_e)
            df = max(1, len(bin_o) - 2)
            p_val = 1 - chi2.cdf(chi2_stat, df)
            
            stats_txt = f"Mean: {sample_mean:.1f}\nGoF p={p_val:.3f}"
            ax.text(0.6, 0.6, stats_txt, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

    except Exception as e:
        print(f"Fit failed: {e}")
    
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_xlabel("Range Size")
    ax.set_ylabel("Count")

if __name__ == "__main__":
    print("=== Integrated DCFTP Metacommunity Simulation ===")
    
    # Check config dispersal settings
    print(f"Dispersal Rate: {config.DISPERSAL_RATE}")
    print(f"Long Distance Prob: {config.LONG_DISTANCE_PROB}")
    if config.DISPERSAL_KERNEL is not None:
        print("Using Custom Dispersal Kernel (FFT enabled)")
    else:
        print("Using Nearest Neighbor Dispersal")

    # --- INITIALIZE BOTH MODELS ---
    print("\nInitializing Models...")
    model_t1 = IntegratedDCFTP(species_type=1, csv_path='metacommunity_fields.csv')
    model_t2 = IntegratedDCFTP(species_type=2, csv_path='metacommunity_fields.csv')
    
    # --- COMBINED TUNING ---
    # Target Mean Inverse Occupancy
    # E.g. target 1/20 = 0.05
    target_inv = 0.1
    optimal_s = tune_combined_extirpation_scaling([model_t1, model_t2], target_inv_mu=target_inv)
    print(f"Optimal Combined Scaling: {optimal_s:.4f}")
    
    # Set scaling for both
    model_t1.set_extirpation_scaling(optimal_s)
    model_t2.set_extirpation_scaling(optimal_s)
    
    # --- GENERATE FINAL SAMPLES ---
    num_samples = 400
    seeds = [secrets.randbits(32) for _ in range(num_samples)]
    
    print("\nGenerating Final Type 1 Samples...")
    samples_t1 = []
    for i, s in enumerate(seeds):
        res = model_t1.generate_single_survivor(s)
        if res is not None and np.any(res): samples_t1.append(res)
    
    print("\nGenerating Final Type 2 Samples...")
    samples_t2 = []
    for i, s in enumerate(seeds):
        res = model_t2.generate_single_survivor(s)
        if res is not None and np.any(res): samples_t2.append(res)

    # --- VISUALIZATION ---
    print("\nGenerating Plots...")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Parameter Visualization (Environmental Gradient)
    rows = np.arange(NUM_PATCHES_Y)
    ax[0,0].plot(rows, model_t1.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], 
                 label='Type 1 MeanB', color='skyblue', marker='o')
    ax[0,0].plot(rows, model_t2.meanB_field.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)[:,0], 
                 label='Type 2 MeanB', color='salmon', marker='x')
    ax[0,0].set_title("Input: Mean Abundance Profile")
    ax[0,0].set_xlabel("Grid Row (Y)")
    ax[0,0].set_ylabel("Biomass")
    ax[0,0].legend()
    ax[0,0].grid(alpha=0.3)

    # 2. Example Spatial Snapshot (Type 1)
    if len(samples_t1) > 0:
        example = samples_t1[0].reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
        ax[0,1].imshow(example, cmap='Blues', interpolation='nearest')
        ax[0,1].set_title(f"Example Range (Type 1)\nScaling: {optimal_s:.2f}")
    else:
        ax[0,1].text(0.5, 0.5, "No Survivors", ha='center')

    # 3. Type 1 Distribution
    analyze_and_plot(ax[1,0], samples_t1, "Type 1 Occupancy Distribution", "skyblue")
    
    # 4. Type 2 Distribution
    analyze_and_plot(ax[1,1], samples_t2, "Type 2 Occupancy Distribution", "salmon")
    
    plt.tight_layout()
    plt.show()
    print("Done.")
