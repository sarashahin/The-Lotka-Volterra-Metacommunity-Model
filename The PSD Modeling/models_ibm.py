

############################################
# models_ibm.py
############################################
"""
Individual-Based Model (IBM) approach for population dynamics.
using binomial & Poisson draws each step.
Now includes multi-patch dynamics with dispersal.
"""
import numpy as np
import logging
from config import (
    BODY_MASS,
    INV,
    MORTALITY_RATE,
    STEP_SIZE,
    TMAX,
    N_RECORDS,
    RECORDING_STEP_SIZE,
    NUM_PATCHES_X,
    NUM_PATCHES_Y,
    DISPERSAL_RATE
)
from dispersal import compute_dispersal

logger = logging.getLogger(__name__)

class IBMModel:
    """
    IBM Model with multi-patch dynamics:
    - N: integer counts for each species in each patch
    - Convert to biomass by N * BODY_MASS
    - Growth rates: r - C@B
    - Includes dispersal between patches
    - Supports both adult and propagule dispersal
    """
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123, dispersal_type='adult'):
        """
        :param r: 1D array of intrinsic growth rates (length S).
        :param C: 2D competition matrix (SxS).
        :param nsteps: number of steps in simulation (default TMAX).
        :param record_step: record every record_step steps.
        :param seed: random seed for reproducibility.
        :param dispersal_type: 'adult' or 'propagule' - specifies which life stage disperses
        """
        self.r = r
        self.C = C
        self.S = len(r)  # number of species
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        self.dispersal_type = dispersal_type

        np.random.seed(seed)

        # Initialize counts (N) for each patch
        # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        init_biomass = BODY_MASS/10
        self.N = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), 
                        int(init_biomass / BODY_MASS), dtype=int)

        # Storage for trajectory
        self.nrecords = self.nsteps // self.record_step
        # Shape: (nrecords, S, NUM_PATCHES_Y, NUM_PATCHES_X)
        self.trajectory = np.full((self.nrecords, self.S, NUM_PATCHES_Y, NUM_PATCHES_X), 
                                 np.nan, dtype=float)

    def run(self):
        """
        Run the IBM simulation with multi-patch dynamics.
        """
        logger.info("Starting IBM simulation with multi-patch dynamics...")
        record_idx = 0
        
        for s in range(self.nsteps):
            # Convert counts to biomass for all patches
            B = self.N * BODY_MASS  # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
            
            # Compute dispersal flux between patches
            outgoing_flux, incoming_flux = compute_dispersal(B)  # Same shape as B
            
            # Reshape B for matrix multiplication with C
            B_reshaped = B.reshape(self.S, -1)  # Shape: (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            
            # Calculate growth rates for all patches at once
            # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            local_growth_rates = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
            
            # Handle fast dying: localGrowthRate < - MORTALITY_RATE
            fast_dying = local_growth_rates < (-MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -MORTALITY_RATE
            
            # Step
            # Death
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            
            # Birth
            birth_lambda = (np.exp((local_growth_rates + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            # Handle dispersal based on type
            if self.dispersal_type == 'adult':
                # Adult dispersal: remove from existing population
                dispersal_prob = outgoing_flux * STEP_SIZE / (B + 1e-10)
                outgoing = np.random.binomial(new_N, dispersal_prob)
                new_N = new_N - outgoing
                incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
                self.N = new_N + birth_values + incoming
            else:  # propagule dispersal
                # Propagule dispersal: remove from new births
                dispersal_prob = outgoing_flux * STEP_SIZE / (B + 1e-10)
                outgoing = np.random.binomial(birth_values, dispersal_prob)
                birth_values = birth_values - outgoing
                incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
                self.N = new_N + birth_values + incoming
            
            # Ensure no negative counts
            self.N = np.maximum(self.N, 0)
            
            # Recording
            if (s+1) % self.record_step == 0:
                self.trajectory[record_idx] = self.N * BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"IBM Progress: {record_idx}/{self.nrecords} records recorded.")

        logger.info("IBM simulation completed.")
        return self.trajectory
      

############################################
# Testing
############################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import correlate
    
    # Modify test parameters for visualization
    def test_ibm_model():
        """
        Comprehensive testing of the IBMModel class.
        Tests dispersal, biomass, growth rates, birth/death, and population dynamics.
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Test parameters
        S = 3  # number of species
        nsteps = 50000  # extended simulation time for better convergence
        
        # Adjust growth rates and competition for more dynamic behavior
        r = np.array([0.8, 0.6, 0.7])  # higher growth rates
        C = np.array([
            [0.2, 0.1, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.2]
        ])
        
        # Calculate analytical equilibrium solution
        print("\nAnalytical Equilibrium Analysis:")
        try:
            C_inv = np.linalg.inv(C)
            B_eq = C_inv @ r
            print(f"Analytical equilibrium biomass: {B_eq}")
            
            # Verify equilibrium conditions
            growth_rates_eq = r - C @ B_eq
            print(f"Growth rates at equilibrium: {growth_rates_eq}")
            print(f"Max absolute growth rate at equilibrium: {np.max(np.abs(growth_rates_eq))}")
            
            # Check if all components are positive
            print(f"All components positive: {np.all(B_eq > 0)}")
            
        except np.linalg.LinAlgError:
            print("Warning: Competition matrix is not invertible")
            B_eq = None
        
        # Test both dispersal types
        for dispersal_type in ['adult', 'propagule']:
            print(f"\nTesting {dispersal_type} dispersal:")
            
            # Initialize model
            model = IBMModel(r=r, C=C, nsteps=nsteps, record_step=10, dispersal_type=dispersal_type)
            
            # Initialize patches with more variation
            base_biomass = BODY_MASS * 100
            model.N = np.zeros((S, NUM_PATCHES_Y, NUM_PATCHES_X), dtype=int)
            for y in range(NUM_PATCHES_Y):
                for x in range(NUM_PATCHES_X):
                    variation = 1 + 0.3 * (np.random.rand(S) - 0.5)
                    model.N[:, y, x] = (base_biomass * variation / BODY_MASS).astype(int)
            
            # Run simulation
            trajectory = model.run()
            
            # Analyze convergence to equilibrium
            if B_eq is not None:
                # Calculate mean biomass across patches and time
                mean_biomass = np.mean(trajectory, axis=(0,2,3))  # average across time and patches
                std_biomass = np.std(trajectory, axis=(0,2,3))
                
                print(f"\nConvergence Analysis for {dispersal_type} dispersal:")
                print(f"Mean final biomass: {mean_biomass}")
                print(f"Standard deviation: {std_biomass}")
                print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
                
                # Check growth rates at final state
                final_growth_rates = r - C @ mean_biomass
                print(f"Final growth rates: {final_growth_rates}")
                print(f"Max absolute growth rate: {np.max(np.abs(final_growth_rates))}")
                
                # Analyze temporal convergence
                time_series = np.mean(trajectory, axis=(2,3))  # average across patches
                growth_rates_series = np.array([r - C @ time_series[i] for i in range(len(time_series))])
                max_growth_rates = np.max(np.abs(growth_rates_series), axis=1)
                
                # Find time to convergence (when growth rates are small)
                convergence_threshold = 0.01
                convergence_time = np.where(max_growth_rates < convergence_threshold)[0]
                if len(convergence_time) > 0:
                    print(f"Time to convergence: {convergence_time[0] * model.record_step} steps")
                else:
                    print("System did not converge within simulation time")
                
                # Detailed Spatial Pattern Analysis
                print("\nSpatial Pattern Analysis:")
                
                # 1. Spatial heterogeneity
                final_spatial_std = np.std(trajectory[-1], axis=(1,2))  # std across patches
                print(f"Spatial heterogeneity: {final_spatial_std}")
                print(f"Relative spatial heterogeneity: {final_spatial_std / mean_biomass}")
                
                # 2. Center vs Edge analysis
                center_y = NUM_PATCHES_Y // 2
                center_x = NUM_PATCHES_X // 2
                center_biomass = np.mean(trajectory[-1, :, center_y-1:center_y+2, center_x-1:center_x+2], axis=(1,2))
                edge_biomass = np.mean([
                    np.mean(trajectory[-1, :, 0, :], axis=1),  # top
                    np.mean(trajectory[-1, :, -1, :], axis=1),  # bottom
                    np.mean(trajectory[-1, :, :, 0], axis=1),  # left
                    np.mean(trajectory[-1, :, :, -1], axis=1)  # right
                ], axis=0)
                print(f"Center vs Edge biomass ratio: {center_biomass/edge_biomass}")
                
                # 3. Spatial autocorrelation (Moran's I)
                # Calculate spatial autocorrelation for each species
                for s in range(S):
                    species_data = trajectory[-1, s, :, :]
                    
                    # Calculate mean and center the data
                    mean_data = np.mean(species_data)
                    centered_data = species_data - mean_data
                    
                    # Create spatial weight matrix (binary connectivity)
                    weights = np.zeros((NUM_PATCHES_Y, NUM_PATCHES_X))
                    for i in range(NUM_PATCHES_Y):
                        for j in range(NUM_PATCHES_X):
                            # Check all 8 neighbors
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < NUM_PATCHES_Y and 0 <= nj < NUM_PATCHES_X:
                                        weights[i, j] += 1
                    
                    # Calculate Moran's I
                    numerator = 0
                    denominator = np.sum(centered_data**2)
                    total_weights = np.sum(weights)
                    
                    for i in range(NUM_PATCHES_Y):
                        for j in range(NUM_PATCHES_X):
                            if weights[i, j] > 0:
                                for di in [-1, 0, 1]:
                                    for dj in [-1, 0, 1]:
                                        if di == 0 and dj == 0:
                                            continue
                                        ni, nj = i + di, j + dj
                                        if 0 <= ni < NUM_PATCHES_Y and 0 <= nj < NUM_PATCHES_X:
                                            numerator += centered_data[i, j] * centered_data[ni, nj]
                    
                    morans_i = (NUM_PATCHES_Y * NUM_PATCHES_X / total_weights) * (numerator / denominator)
                    print(f"Species {s} Moran's I: {morans_i:.3f}")
                
                # 4. Patch connectivity analysis
                # Calculate correlation between adjacent patches
                center_values = []
                neighbor_values = []
                final_state = trajectory[-1]  # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
                
                # Calculate total biomass for each patch
                patch_total_biomass = np.sum(final_state, axis=0)  # Sum across species dimension
                
                for y in range(NUM_PATCHES_Y):
                    for x in range(NUM_PATCHES_X):
                        # Get valid neighboring patches
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < NUM_PATCHES_Y and 0 <= nx < NUM_PATCHES_X:
                                    center_values.append(patch_total_biomass[y, x])
                                    neighbor_values.append(patch_total_biomass[ny, nx])
                
                if len(center_values) >= 2:
                    corr, p_value = stats.pearsonr(center_values, neighbor_values)
                    print(f"\nSpatial correlation analysis:")
                    print(f"Mean adjacent patch correlation: {corr:.3f} (p={p_value:.3f})")
                
                # Calculate distance-based correlations
                print("\nDistance-based correlation analysis:")
                for distance in range(1, 4):
                    center_values = []
                    neighbor_values = []
                    for i in range(NUM_PATCHES_Y):
                        for j in range(NUM_PATCHES_X):
                            # Check all 8 neighbors at this distance
                            for di in [-distance, 0, distance]:
                                for dj in [-distance, 0, distance]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < NUM_PATCHES_Y and 0 <= nj < NUM_PATCHES_X:
                                        center_values.append(patch_total_biomass[i, j])
                                        neighbor_values.append(patch_total_biomass[ni, nj])
                    
                    if len(center_values) >= 2:
                        corr, p_value = stats.pearsonr(center_values, neighbor_values)
                        print(f"Correlation at distance {distance}: {corr:.3f} (p={p_value:.3f})")
                    else:
                        print(f"Not enough valid pairs found at distance {distance}")
        
        return trajectory

    # Run the tests
    trajectory = test_ibm_model()
    

def test_dispersal_conservation():
    """Test that total biomass is conserved during dispersal"""
    B = np.random.rand(S, NUM_PATCHES_Y, NUM_PATCHES_X)  # Random initial biomass
    outgoing, incoming = compute_dispersal(B)
    # Total outgoing should equal total incoming
    assert np.allclose(np.sum(outgoing), np.sum(incoming))
    # Neither should exceed total biomass
    assert np.all(outgoing <= B * DISPERSAL_RATE)

def test_isolated_patch():
    """Test behavior of an isolated high-biomass patch"""
    B = np.zeros((S, NUM_PATCHES_Y, NUM_PATCHES_X))
    B[:, 5, 5] = 10.0
    # Create simple test parameters
    r = np.array([0.5])
    C = np.array([[0.1]])
    model = IBMModel(r=r, C=C, nsteps=100)
    model.N = (B / BODY_MASS).astype(int)
    trajectory = model.run()
    # Check if dispersal creates expected spatial pattern
    # and if total biomass is conserved
    return trajectory
