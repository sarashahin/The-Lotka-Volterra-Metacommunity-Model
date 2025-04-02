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
import matplotlib.pyplot as plt


from config_base import IBMConfig
from dispersal import compute_dispersal, generate_habitat_quality, compute_dispersal_matrix
from visualization import (
    plot_spatial_patterns, plot_dispersal_patterns, plot_temporal_evolution,
    plot_environmental_effects, plot_seed_size_effects, plot_species_comparison,
    plot_dispersal_analysis, plot_all_analyses
)
from real_species_vis import (
    plot_real_species_distribution, plot_dispersal_mechanisms,
    plot_habitat_suitability
)

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
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123, dispersal_type='adult', config=None):
        """
        :param r: 1D array of intrinsic growth rates (length S).
        :param C: 2D competition matrix (SxS).
        :param nsteps: number of steps in simulation (default TMAX).
        :param record_step: record every record_step steps.
        :param seed: random seed for reproducibility.
        :param dispersal_type: 'adult' or 'propagule' - specifies which life stage disperses
        :param config: IBMConfig object for configuration parameters
        """
        self.r = r
        self.C = C
        self.S = len(r)  # number of species
        self.config = config if config is not None else IBMConfig()
        self.nsteps = nsteps if nsteps is not None else self.config.TMAX
        self.record_step = record_step if record_step is not None else self.config.RECORDING_STEP_SIZE
        self.dispersal_type = dispersal_type

        np.random.seed(seed)

        # Initialize with higher biomass and spatial variation
        init_biomass = self.config.BODY_MASS * 10  # Reduced initial biomass
        self.N = np.zeros((self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X), dtype=int)
        
        # Add spatial variation in initial conditions
        for s in range(self.S):
            for y in range(self.config.NUM_PATCHES_Y):
                for x in range(self.config.NUM_PATCHES_X):
                    # Random variation around init_biomass
                    variation = 1 + 0.5 * (np.random.rand() - 0.5)
                    self.N[s, y, x] = int(init_biomass * variation / self.config.BODY_MASS)

        # Storage for trajectory
        self.nrecords = self.nsteps // self.record_step
        # Shape: (nrecords, S, NUM_PATCHES_Y, NUM_PATCHES_X)
        self.trajectory = np.full((self.nrecords, self.S, self.config.NUM_PATCHES_Y, self.config.NUM_PATCHES_X), 
                                 np.nan, dtype=float)
        
        # Pre-compute dispersal matrices
        self.dispersal_matrices = compute_dispersal_matrix(self.config)

    def run(self):
        """Run simulation with enhanced monitoring"""
        print(f"\nRunning {self.dispersal_type} dispersal simulation...")
        record_idx = 0
        
        for s in range(self.nsteps):
            # Convert counts to biomass
            B = self.N * self.config.BODY_MASS
            
            # Compute dispersal using pre-computed matrices
            outgoing_flux, incoming_flux = compute_dispersal(B, self.config, self.dispersal_matrices)
            
            # Calculate growth rates
            B_reshaped = B.reshape(self.S, -1)
            local_growth_rates = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
            
            # Handle mortality with reduced effect
            fast_dying = local_growth_rates < (-self.config.MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, self.config.MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -self.config.MORTALITY_RATE
            
            # Population dynamics with adjusted rates
            survival_prob = np.exp(-full_mortality * self.config.STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            birth_lambda = (np.exp((local_growth_rates + self.config.MORTALITY_RATE) * self.config.STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            # Handle dispersal based on type with reduced rates
            if self.dispersal_type == 'adult':
                dispersal_prob = outgoing_flux * self.config.STEP_SIZE / (B + 1e-10)
                outgoing = np.random.binomial(new_N, dispersal_prob)
                new_N = new_N - outgoing
                incoming = np.random.poisson(incoming_flux * self.config.STEP_SIZE / self.config.BODY_MASS)
                self.N = new_N + birth_values + incoming
            else:  # propagule dispersal
                dispersal_prob = outgoing_flux * self.config.STEP_SIZE / (B + 1e-10)
                outgoing = np.random.binomial(birth_values, dispersal_prob)
                birth_values = birth_values - outgoing
                incoming = np.random.poisson(incoming_flux * self.config.STEP_SIZE / self.config.BODY_MASS)
                self.N = new_N + birth_values + incoming
            
            # Ensure non-negative populations
            self.N = np.maximum(self.N, 0)
            
            # Recording
            if (s+1) % self.record_step == 0:
                self.trajectory[record_idx] = self.N * self.config.BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    print(f"Progress: {record_idx}/{self.nrecords} records")
                    # Print mean biomass for monitoring
                    mean_biomass = np.mean(self.N * self.config.BODY_MASS, axis=(1,2))
                    print(f"Current mean biomass: {mean_biomass}")
        
        return self.trajectory

def test_ibm_model():
    """Comprehensive testing with enhanced visualization"""
    # Test parameters with higher growth rates and lower competition
    S = 3  # number of species
    r = np.array([1.2, 1.0, 1.1])  # growth rates
    C = np.array([
        [0.1, 0.05, 0.05],  #  competition
        [0.05, 0.1, 0.05],
        [0.05, 0.05, 0.1]
    ])
    
    # Define seed sizes for each species
    seed_sizes = np.array([1.0, 0.5, 0.8])  # Example seed sizes
    
    # Analytical equilibrium analysis
    print("\nAnalytical Equilibrium Analysis:")
    C_inv = np.linalg.inv(C)
    B_eq = C_inv @ r
    print(f"Analytical equilibrium biomass: {B_eq}")
    growth_rates_eq = r - C @ B_eq
    print(f"Growth rates at equilibrium: {growth_rates_eq}")
    
    # Generate habitat quality
    config = IBMConfig()
    habitat_quality = generate_habitat_quality()
    
    # Run simulations for both dispersal types
    results = {}
    for dispersal_type in ['adult', 'propagule']:
        # Initialize and run model with shorter simulation time
        model = IBMModel(r=r, C=C, nsteps=10000, record_step=100, 
                        dispersal_type=dispersal_type, config=config)
        trajectory = model.run()
        results[dispersal_type] = trajectory
        
        # Basic analysis
        print(f"\nAnalysis for {dispersal_type} dispersal:")
        mean_biomass = np.mean(trajectory[-1], axis=(1,2))
        print(f"Mean final biomass: {mean_biomass}")
        print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
        
        # Generate all plots
        plot_all_analyses(trajectory, dispersal_type, config.WIND_DIRECTION, 
                         config.WIND_STRENGTH, habitat_quality, seed_sizes)
        
        # Generate real species visualizations
        plot_real_species_distribution(trajectory, f'output_{dispersal_type}')
        plot_dispersal_mechanisms(trajectory, f'output_{dispersal_type}')
        plot_habitat_suitability(trajectory, f'output_{dispersal_type}')
    
    return results

def test_dispersal_conservation():
    """Test that total biomass is conserved during dispersal"""
    config = IBMConfig()
    S = 3  # number of species
    B = np.random.rand(S, config.NUM_PATCHES_Y, config.NUM_PATCHES_X)  # Random initial biomass
    outgoing, incoming = compute_dispersal(B)
    # Total outgoing should equal total incoming
    assert np.allclose(np.sum(outgoing), np.sum(incoming))
    # Neither should exceed total biomass
    assert np.all(outgoing <= B * config.DISPERSAL_RATE)

def test_isolated_patch():
    """Test behavior of an isolated high-biomass patch"""
    config = IBMConfig()
    S = 1  # single species for this test
    B = np.zeros((S, config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    B[:, 5, 5] = 10.0
    # Create simple test parameters
    r = np.array([0.5])
    C = np.array([[0.1]])
    model = IBMModel(r=r, C=C, nsteps=100, config=config)
    model.N = (B / config.BODY_MASS).astype(int)
    trajectory = model.run()
    # Check if dispersal creates expected spatial pattern
    # and if total biomass is conserved
    return trajectory

def validate_model_assumptions():
    """Validate key model assumptions and behaviors with detailed equilibrium analysis"""
    # Test parameters
    S = 3
    r = np.array([1.2, 1.0, 1.1])
    C = np.array([
        [0.1, 0.05, 0.05],
        [0.05, 0.1, 0.05],
        [0.05, 0.05, 0.1]
    ])
    
    # 1. Analytical equilibrium calculation
    print("\nAnalytical Equilibrium Analysis:")
    C_inv = np.linalg.inv(C)
    B_eq = C_inv @ r
    print(f"Analytical equilibrium biomass: {B_eq}")
    
    # 2. Run simulation with different dispersal rates
    dispersal_rates = [0.1, 0.2, 0.3]  # Test multiple rates
    print("\nTesting equilibrium convergence at different dispersal rates:")
    
    for rate in dispersal_rates:
        # Create new config with modified dispersal rate
        config = IBMConfig()
        config.DISPERSAL_RATE = rate
        
        model = IBMModel(r=r, C=C, nsteps=5000, config=config)
        trajectory = model.run()
        final_state = trajectory[-1]
        mean_biomass = np.mean(final_state, axis=(1,2))
        
        # Check dt/dB = B(r−CB) ≈ 0
        growth_rates = r - C @ mean_biomass
        net_growth = mean_biomass * growth_rates
        
        print(f"\nDispersal rate: {rate}")
        print(f"Mean biomass: {mean_biomass}")
        print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
        print(f"Net growth rates (should be ≈ 0): {net_growth}")
        print(f"Growth rates (r-CB): {growth_rates}")
        
        # Check if B > 0 assumption holds
        min_biomass = np.min(final_state)
        print(f"Minimum biomass: {min_biomass}")
    
    # 3. Test continuous approximation
    print("\nTesting continuous approximation:")
    step_sizes = [0.1, 0.05, 0.01]
    for dt in step_sizes:
        config = IBMConfig()
        config.STEP_SIZE = dt
        model = IBMModel(r=r, C=C, nsteps=int(5000 * 0.1/dt), config=config)
        trajectory = model.run()
        final_state = trajectory[-1]
        mean_biomass = np.mean(final_state, axis=(1,2))
        print(f"\nStep size: {dt}")
        print(f"Mean biomass: {mean_biomass}")
        print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
    
    # 4. Spatial homogeneity analysis
    print("\nSpatial homogeneity analysis:")
    final_state = trajectory[-1]
    spatial_var = np.var(final_state, axis=(1,2))
    spatial_cv = spatial_var / np.mean(final_state, axis=(1,2))
    print(f"Spatial coefficient of variation: {spatial_cv}")
    
    # 5. Test equilibrium stability
    print("\nTesting equilibrium stability:")
    # Run with perturbed initial conditions
    config = IBMConfig()
    model = IBMModel(r=r, C=C, nsteps=5000, config=config)
    init_biomass = B_eq.reshape(-1, 1, 1) * (1 + 0.1 * np.random.randn(S, config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    model.N = (init_biomass / config.BODY_MASS).astype(int)
    trajectory = model.run()
    final_state = trajectory[-1]
    mean_biomass = np.mean(final_state, axis=(1,2))
    print(f"Final biomass with perturbed initial conditions: {mean_biomass}")
    print(f"Relative error from equilibrium: {(mean_biomass - B_eq) / B_eq}")
    
    return True

if __name__ == "__main__":
    # Run validation first
    print("Running model validation...")
    validate_model_assumptions()
    
    # Then run main simulation
    print("\nRunning main simulation...")
    results = test_ibm_model()
