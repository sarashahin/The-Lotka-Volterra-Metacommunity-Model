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
from dispersal import LOCAL_DISPERSAL_MATRIX


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
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123, dispersal_type='propagule', dispersal_away_rate=None):
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
        
        if dispersal_away_rate is not None:
            self.dispersal_away_rate = dispersal_away_rate
        else:
            self.dispersal_away_rate = \
                np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten(). \
                reshape((NUM_PATCHES_Y, NUM_PATCHES_X))

        np.random.seed(seed)

        # Initialize counts (N) for each patch
        # Shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        init_biomass = BODY_MASS /10
        self.N = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), 
                        max(1, int(init_biomass / BODY_MASS)), dtype=int)

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
            
            if s%1==0:
                print(f"step {s}, {s/self.nsteps}, {np.sum(self.N)}")
                
            # Compute dispersal flux between patches
            incoming_flux = compute_dispersal(B)  # Same shape as B
            
            # Reshape B for matrix multiplication with C
            B_reshaped = B.reshape(self.S, -1)  # Shape: (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            
            # Calculate growth rates for all patches at once
            # C @ B_reshaped will have shape (S, NUM_PATCHES_Y * NUM_PATCHES_X)
            local_growth_rates = (self.r.reshape(-1, 1) - self.C @ B_reshaped).reshape(B.shape)
            if self.dispersal_type == 'adult':
                local_growth_rates = local_growth_rates - \
                    np.broadcast_to(self.dispersal_away_rate,local_growth_rates.shape)
            
            # Handle fast dying: localGrowthRate < - MORTALITY_RATE
            fast_dying = local_growth_rates < (-MORTALITY_RATE)
            full_mortality = np.full_like(local_growth_rates, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rates[fast_dying]
            local_growth_rates[fast_dying] = -MORTALITY_RATE
            
            # For adult dispersal, add the dispersal-away rate to mortality here.
            if self.dispersal_type == 'adult':
                full_mortality = full_mortality + self.dispersal_away_rate
            
            # Step
            # Death
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)
            
            # Birth
            birth_lambda = (np.exp((local_growth_rates + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)
            
            # Incoming dispersers are computed from the incoming_flux.
            incoming = np.random.poisson(incoming_flux * STEP_SIZE / BODY_MASS)
            # Update population by adding surviving individuals, new births, and incoming dispersers.
            self.N = new_N + birth_values + incoming 
            
            # Ensure no negative counts
            if (self.N < 0).any():
                sys.exit("Negative abundances!!")
            
            # Recording
            if (s+1) % self.record_step == 0:
                self.trajectory[record_idx] = self.N * BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"IBM Progress: {record_idx}/{self.nrecords} records recorded.")

        logger.info("IBM simulation completed.")
        return self.trajectory
      

############################################
# Testing IBM Model – Updated Test Code
############################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def test_ibm_model_updated():
        """
        Updated test for IBMModel.
        This version:
          - Computes the analytical equilibrium.
          - Runs the IBM simulation.
          - Computes the mean and variance time series of biomass (averaged across patches).
          - Plots the time series for one species.
          - Prints the final mean biomass and the relative error from equilibrium.
        """
        np.random.seed(42)
        
        # Test parameters
        S = 3  # number of species
        nsteps = 1000 # shorter simulation time for testing
        r = np.array([1.0, 1.0, 1.0])
        C = np.array([
            [1.0, 1.7, 0.4],
            [0.4, 1.0, 1.7],
            [1.7, 0.4, 1.0]
        ])
        
        # Analytical equilibrium solution:
        print("\nAnalytical Equilibrium Analysis:")
        try:
            C_inv = np.linalg.inv(C)
            B_eq = C_inv @ r
            print(f"Analytical equilibrium biomass: {B_eq}")
            growth_rates_eq = r - C @ B_eq
            print(f"Growth rates at equilibrium: {growth_rates_eq}")
            print(f"Max absolute growth rate at equilibrium: {np.max(np.abs(growth_rates_eq))}")
            print(f"All components positive: {np.all(B_eq > 0)}")
        except np.linalg.LinAlgError:
            print("Warning: Competition matrix is not invertible")
            B_eq = None
        
        # Initialize IBMModel with a shorter simulation time for testing
        model = IBMModel(r=r, C=C, nsteps=nsteps, record_step=10, dispersal_type="propagule")
        
        # Run simulation. The trajectory is an array with shape (n_records, S, NUM_PATCHES_Y, NUM_PATCHES_X)
        trajectory = model.run()
        
        # Compute the number of records and a time vector (assuming constant record_step timing)
        nrecords = trajectory.shape[0]
        record_times = np.linspace(0, nsteps, nrecords)
        
        # Compute time series of mean and variance (averaged over patches) for each species
        # Here we average over patches (axes 2 and 3)
        mean_time_series = np.mean(trajectory, axis=(2, 3))  # shape: (n_records, S)
        var_time_series  = np.var(trajectory, axis=(2, 3))    # shape: (n_records, S)
        
        # Plot the time series for the first species
        species = 0
        plt.figure(figsize=(8, 5))
        plt.plot(record_times, mean_time_series[:, species], label=f'Mean biomass (Species {species})')
        std_dev = np.sqrt(var_time_series[:, species])
        plt.fill_between(record_times,
                         mean_time_series[:, species] - std_dev,
                         mean_time_series[:, species] + std_dev,
                         color='blue', alpha=0.2, label='Std. Deviation')
        plt.xlabel("Time (steps)")
        plt.ylabel("Biomass")
        plt.title(f"Time Series of Mean Biomass and Std. Dev.IBM Model (Species {species})")
        plt.legend()
        plt.show()
        
        # Compute final mean biomass by averaging over patches at the final recorded time
        final_biomass = trajectory[-1, :, :, :]  # final time slice: shape (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        mean_final = np.mean(final_biomass, axis=(1, 2))  # one mean per species
        print("\nFinal Mean biomass for each species:", mean_final)
        
        if B_eq is not None:
            # Compute relative error from analytical equilibrium
            rel_error = (mean_final - B_eq) / B_eq
            print("Relative error from equilibrium:", rel_error)
    
    test_ibm_model_updated()


                
