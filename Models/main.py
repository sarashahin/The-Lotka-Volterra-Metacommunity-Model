
###############################################################################
# PSD Scheme for multi patches
################################################################################

import os
import numpy as np
import random
import time
import sys
import scipy.io as sio
from pathlib import Path
# from scipy.integrate import solve_ivp
from datetime import datetime
import signal
import traceback
import logging
import cProfile

# Import Assimulo classes for ODE solving
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from metacommunity import Metacommunity  # Assuming the Metacommunity class and related classes are already implemented.
from lvmcm_rng import LVMCM_rng
from topography import Topography
from species import Species

# Configure logging for debugging
logging.basicConfig(level=logging.INFO,  # Set to INFO to reduce verbosity
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("simulation.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

# Constants for configuration
FIX_SEED = 42
ASSEMBLE = True
OUTPUT = True
SNAPSHOT = True
SOURCE_SINK = False
CMAT_REG = False
TRAJECTORY = False
FLUCTUATE = False
WARMING = False
LONGDISTDISP = False
NODE_REMOVAL = False

# New Constants to Prevent Overproduction
MAX_SPECIES = 2000       # Maximum total species allowed
MAX_PRODUCERS = 1500     # Maximum number of producers allowed
MAX_ITERATIONS = 120     # Maximum number of assembly iterations
MAX_CONSUMERS = 500      # Maximum number of consumers allowed

# Controlled invasion rates
INVADERS_PER_ITERATION = {
    0: 1,  # Producers
    1: 1   # Consumers
}


def set_random_seed(fixed_seed=0):
    if fixed_seed > 0:
        g_seed = 1
    else:
        g_seed = random.randint(1, 1000000)
    
    rng_instance = LVMCM_rng()
    rng_instance.set_seed(g_seed)  # Initialize with a specific seed
    return g_seed

def main():
    try:
        # Start the timer
        start_time = time.time()
        print("Starting metacommunity simulation...")

        # Initialize simulation parameters
        a_init = True
        a_invMax = 1000
        a_tMax = 500
        a_outputDirectory = "output/"

        # Ensure output directory exists
        Path(a_outputDirectory).mkdir(parents=True, exist_ok=True)

        # Initialize Topography
        topo = Topography(
            no_nodes=32,
            lattice_height=5,
            lattice_width=5,
            phi=0.3,
            envVar=1,
            skVec=np.array([0.1]),
            var_e=0.5,
            randGraph=False,
            gabriel=True,
            T_int=25.0,
            network_file="",
            scVec=np.array([0.05]) 
        )
        
        # Explicitly generate the network
        topo.gen_network()

        # Initialize Species
        spp = Species(
            topo=topo,
            c1=0.5,
            c2=0.3,
            c3=0.1,
            emRate=0.2,
            dispL=0.6,
            pProducer=0.4,
            prodComp=True,
            symComp=True,
            alpha=0.02,
            sigma=1.5,
            sigma_t=0.2,
            rho=0.1,
            comp_dist=1,
            omega=0.9,
            dispNorm=1.0
        )

        # Generate dispersal matrix
        spp.gen_disp_mat()
        print("Dispersal matrix generated successfully.")

        # Add a producer and a consumer
        spp.invade(0)  # Invade with a producer
        print("Producer added successfully.")
        spp.invade(1)  # Invade with a consumer
        print("Consumer added successfully.")

        # Set random seed for reproducibility
        g_seed = set_random_seed(FIX_SEED)
        print(f"Random Seed Set: {g_seed}")

        # Initialize Metacommunity
        meta = Metacommunity(
            spp=spp,
            a_init=a_init,
            a_bMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
            a_xMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
            a_scMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1params0.mat",
            a_invMax=a_invMax,
            a_tMax=a_tMax,
            a_outputDirectory=a_outputDirectory,
            a_c1=0.5,
            a_c2=0.3,
            a_c3=0.1,
            a_emRate=0.2,
            a_dispL=0.6,
            a_pProducer=0.4,
            a_prodComp=True,
            a_symComp=True,
            a_alpha=0.02,
            a_sigma=1.5,
            a_sigma_t=0.3,
            a_rho=0.1,
            a_comp_dist=1,
            a_omega=0.9,
            a_dispNorm=1.0,
            a_no_nodes=32,
            a_lattice_height=5,
            a_lattice_width=5,
            a_phi=0.3,
            a_envVar=1,
            a_skVec=np.array([0.1]),
            a_var_e=0.5,
            a_randGraph=False,
            a_gabriel=True,
            a_T_int=25.0,
            a_envMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1rMat0.mat",
            a_parOut=1.0,
            a_experiment="/Users/sarashahin/Desktop/model/SimulationData/autonomous_turnover_example_pars.txt",
            a_rep=1,
            storeTraj=1,
            bMatFileName="/Users/sarashahin/Desktop/model/output",
            g_block_transitions=False
        )


        # Print parameters for verification
        meta.print_params()

        # Generate topography/environment
        if a_init:
            meta.spp.topo.gen_landscape()
            print("Landscape generated successfully.")
            meta.spp.gen_disp_mat()
        else:
            meta.spp.topo.gen_landscape(meta.spp.topo.network)
            print("Landscape generated successfully.")

        # Run the assembly process
        if ASSEMBLE:
            assemble_metacommunity(meta)

        # Perform final bookkeeping
        if OUTPUT:
            final_bookkeeping(meta)

        # End the simulation
        elapsed_time = time.time() - start_time
        print(f"Simulation completed successfully in {elapsed_time / 60:.2f} minutes.")

    except Exception as e:
        print("An error occurred during the simulation:")
        traceback.print_exc()


def assemble_metacommunity(meta):
    """
    Assemble the metacommunity by introducing invaders, simulating dynamics, 
    and removing extinct species in each iteration.
    """
    try:
        set_random_seed(FIX_SEED)
        logger.info("Assembly process started.")
        
        # Initialize lists to store trajectory data
        trajectory_producers = []
        trajectory_consumers = []
        trajectory_iterations = []
        trajectory_xMat = []  # To store species abundances per node
        trajectory_rMat = []  # To store species growth rates per node

        max_species = meta.spp.xMat.shape[0] # Initialize with initial number of species
        
        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"Starting iteration {iteration} of assembly...")

            # Check for maximum species limit
            current_species = meta.spp.S_p + meta.spp.S_c  # Calculate total species
            current_producers = meta.spp.S_p
            current_consumers = meta.spp.S_c
            
            # Update max_species if needed
            if current_species > max_species:
                max_species = current_species
            
            trajectory_iterations.append(iteration)
            trajectory_producers.append(current_producers)
            trajectory_consumers.append(current_consumers)
            trajectory_xMat.append(meta.spp.xMat.copy())  # xMat holds species abundances per node
            trajectory_rMat.append(meta.spp.rMat.copy())  # rMat holds species growth rates per node

            if current_species >= MAX_SPECIES:
                print(f"Maximum species limit of {MAX_SPECIES} reached. Stopping assembly.")
                break

            # Check for maximum producers
            if current_producers >= MAX_PRODUCERS:
                logger.info(f"Maximum producers limit of {MAX_PRODUCERS} reached.")
                producers_to_invade = 0
            else:
                producers_to_invade = INVADERS_PER_ITERATION.get(0, 0)
                # Ensure we don't exceed MAX_SPECIES
                producers_to_invade = min(producers_to_invade, MAX_SPECIES - current_species)

            # Check for maximum consumers
            if current_consumers >= MAX_CONSUMERS:
                logger.info(f"Maximum consumers limit of {MAX_CONSUMERS} reached.")
                consumers_to_invade = 0
            else:
                consumers_to_invade = INVADERS_PER_ITERATION.get(1, 0)
                # Ensure we don't exceed MAX_SPECIES
                consumers_to_invade = min(consumers_to_invade, MAX_SPECIES - current_species)

            # Invade producers if allowed
            if producers_to_invade > 0:
                logger.info(f"Invading {producers_to_invade} producer(s).")
                meta.invader_sample(0, producers_to_invade)
            else:
                logger.info("No producers invaded this iteration.")

            # Invade consumers if allowed
            if consumers_to_invade > 0:
                logger.info(f"Invading {consumers_to_invade} consumer(s).")
                meta.invader_sample(1, consumers_to_invade)
            else:
                logger.info("No consumers invaded this iteration.")

            # Simulate Dynamics with Assimulo
            logger.info("Simulating dynamics...")
            meta.meta_c_dynamics(meta.tMax)

            # Extinction Step
            logger.info("Removing extinct species...")
            meta.spp.extinct()
            
            # Record current state after dynamics and extinction
            current_producers = meta.spp.S_p
            current_consumers = meta.spp.S_c
            trajectory_iterations.append(iteration + 0.5)  # To represent state after dynamics
            trajectory_producers.append(current_producers)
            trajectory_consumers.append(current_consumers)
            trajectory_xMat.append(meta.spp.xMat.copy())
            trajectory_rMat.append(meta.spp.rMat.copy())
            
            # Update max_species if needed
            if current_species > max_species:
                max_species = current_species

            # Stop condition: All invasions complete or all species extinct
            if meta.spp.invasion >= meta.invMax:
                print("Maximum invasions reached. Stopping assembly.")
                break
            if meta.spp.S_p == 0 and meta.spp.S_c == 0:
                print("All species extinct. Stopping assembly.")
                break

            print(f"Iteration {iteration} complete.\n")

        else:
            print(f"Reached maximum iterations ({MAX_ITERATIONS}) without meeting stop conditions.")
            
        
        # Determine the maximum number of species across all iterations
        max_species = max(max_species, max(trajectory_producers) + max(trajectory_consumers))

        # Pad xMat matrices to have the same number of species
        for i in range(len(trajectory_xMat)):
            current_xMat = trajectory_xMat[i]
            current_num_species, num_nodes = current_xMat.shape
            if current_num_species < max_species:
                # Create a padded xMat with zeros for new species
                padding = np.zeros((max_species - current_num_species, num_nodes))
                trajectory_xMat[i] = np.vstack((current_xMat, padding))
            elif current_num_species > max_species:
                # This shouldn't happen, but handle just in case
                padding = np.zeros((current_num_species - max_species, num_nodes))
                trajectory_xMat[i] = np.vstack((trajectory_xMat[i], padding))
        
        # Pad rMat matrices to have the same number of species
        for i in range(len(trajectory_rMat)):
            current_rMat = trajectory_rMat[i]
            current_num_species, num_nodes = current_rMat.shape
            if current_num_species < max_species:
                # Create a padded rMat with zeros for new species
                padding = np.zeros((max_species - current_num_species, num_nodes))
                trajectory_rMat[i] = np.vstack((current_rMat, padding))
            elif current_num_species > max_species:
                # This shouldn't happen, but handle just in case
                padding = np.zeros((current_num_species - max_species, num_nodes))
                trajectory_rMat[i] = np.vstack((trajectory_rMat[i], padding))
        
        # Convert the lists to 3D NumPy arrays
        trajectory_xMat = np.array(trajectory_xMat)
        trajectory_rMat = np.array(trajectory_rMat)

        # Save trajectory data to a file for visualization
        trajectory_data = {
            'iteration': np.array(trajectory_iterations),
            'producers': np.array(trajectory_producers),
            'consumers': np.array(trajectory_consumers),
            'xMat': trajectory_xMat,  # Shape: (num_records, max_species, nodes)
            'rMat': trajectory_rMat   # Shape: (num_records, max_species, nodes)
        }
        trajectory_file = os.path.join(meta.outputDirectory, 'trajectory_data.npz')
        np.savez(trajectory_file, **trajectory_data)
        logger.info(f"Trajectory data saved to {trajectory_file}")

    except Exception as e:
        print("An error occurred during the assembly process:")
        traceback.print_exc()


def final_bookkeeping(meta):
    """
    Perform final bookkeeping steps after the simulation.
    """
    try:
        logger.info("Final bookkeeping...")
        meta.meta_c_dynamics(1000)  # Final dynamics relaxation
        meta.spp.extinct()  # Remove extinct species
        meta.saveMC()
        logger.info("Bookkeeping complete.")
    except Exception as e:
        logger.error("An error occurred during final bookkeeping:")
        logger.error(traceback.format_exc())


#  meta_c_dynamics method to use Assimulo

def meta_c_dynamics(self, t_end):
    """
    Integrate the metacommunity dynamics up to time t_end using Assimulo.
    """
    try:
        # Initial state
        y0 = self.spp.xMat.flatten()

        def rhs(t, y):
            # Reshape state
            state = y.reshape(self.spp.xMat.shape)
            # meta.dynamics(t, state) should return dX/dt as a 2D array matching xMat
            dXdt = self.dynamics(t, state)
            return dXdt.flatten()

        problem = Explicit_Problem(rhs, y0, 0.0)
        solver = CVode(problem)
        # Set solver options as needed
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.maxsteps = 10000

        # Integrate until t_end
        solver.simulate(t_end)

        # Update state at the end of integration
        self.spp.xMat = solver.y.reshape(self.spp.xMat.shape)
        logger.debug(f"Dynamics simulated up to time {t_end}.")
        
        # **Compute and Update rMat Based on the New xMat**
        self.spp.rMat = self.spp.compute_intrinsic_growth_rates(self.spp.xMat)
        logger.debug(f"rMat updated based on new xMat.")
        
        
    except Exception as e:
        logger.error("An error occurred during dynamics simulation:")
        logger.error(traceback.format_exc())

# Attach the meta_c_dynamics method to Metacommunity class if not already present
if not hasattr(Metacommunity, 'meta_c_dynamics'):
    setattr(Metacommunity, 'meta_c_dynamics', meta_c_dynamics)

if __name__ == "__main__":
    # Profile the main function and save the profile to 'simulation_profile.prof'
    cProfile.run('main()', 'simulation_profile.prof')




##########################################################################
# PSD scheme for smaller patch
##########################################################################
import os
import numpy as np
import time
import traceback
import matplotlib.pyplot as plt
from pathlib import Path
from metacommunity import Metacommunity  #  Metacommunity class is implemented
from topography import Topography
from species import Species

# Constants for configuration
FIX_SEED = 42
OUTPUT_DIR = "output_figures/"

def set_random_seed(fixed_seed=0):
    if fixed_seed > 0:
        np.random.seed(fixed_seed)
    else:
        np.random.seed(np.random.randint(1, 1000000))

def classify_outcome(m1, m2, alpha):
    """
    Classify outcomes based on ecological rules:
    - "1 wins" if m1 dominates
    - "2 wins" if m2 dominates
    - "Coexist" if conditions favor coexistence
    - "Founder control" for intermediate cases
    """
    if m1 > m2 * alpha:
        return 1  # "1 wins"
    elif m2 > m1 * alpha:
        return 2  # "2 wins"
    elif abs(m1 - m2) < alpha * 0.1:
        return 3  # "Coexist"
    else:
        return 4  # "Founder control"

def simulate_psd_scheme(alphas, m1_range, m2_range):
    """
    Test the PSD scheme by reproducing Figures 10 and 11.
    """
    results = {alpha: np.zeros((len(m1_range), len(m2_range))) for alpha in alphas}

    for alpha in alphas:
        for i, m1 in enumerate(m1_range):
            for j, m2 in enumerate(m2_range):
                results[alpha][i, j] = classify_outcome(m1, m2, alpha)

    return results

def plot_results(results, alphas, m1_range, m2_range):
    """
    Visualize the simulation results to match Figures 10 and 11.
    """
    cmap = plt.cm.get_cmap("tab10", 4)  # Custom colormap
    region_labels = ["1 wins", "2 wins", "Coexist", "Founder control"]

    fig, axes = plt.subplots(1, len(alphas), figsize=(15, 5))

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        im = ax.imshow(
            results[alpha],
            origin='lower',
            extent=(np.log10(m1_range[0]), np.log10(m1_range[-1]),
                    np.log10(m2_range[0]), np.log10(m2_range[-1])),
            aspect='auto',
            cmap=cmap,
            vmin=1,
            vmax=4
        )
        ax.set_title(f"\u03B1 = {alpha}")
        ax.set_xlabel("log10(m1)")
        ax.set_ylabel("log10(m2)")

    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(region_labels)
    cbar.set_label("Outcome")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_10_11_test.png"))
    plt.show()

def main():
    try:
        # Start the timer
        start_time = time.time()
        print("Starting PSD scheme testing...")

        # Ensure output directory exists
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        # Parameter ranges for Figures 10 and 11
        alphas = [0.9, 0.99, 1.0, 1.1, 1.5]  # α values
        m1_range = np.logspace(-3, 2, 50)  # m1 values
        m2_range = np.logspace(-3, 2, 50)  # m2 values

        # Run simulation
        results = simulate_psd_scheme(alphas, m1_range, m2_range)

        # Visualize results
        plot_results(results, alphas, m1_range, m2_range)

        # End the simulation
        elapsed_time = time.time() - start_time
        print(f"Simulation completed successfully in {elapsed_time / 60:.2f} minutes.")

    except Exception as e:
        print("An error occurred during the simulation:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
