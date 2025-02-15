
import os
import numpy as np
import random
import time
import sys
import scipy.io as sio
from pathlib import Path
from datetime import datetime
import signal
import traceback
import logging
import cProfile

# Import Assimulo classes for ODE solving
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from metacommunity import Metacommunity  #  Metacommunity class and related classes
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
MAX_PRODUCERS = 10000     # Maximum number of producers allowed
MAX_ITERATIONS = 100     # Maximum number of assembly iterations
MAX_CONSUMERS = 8000      # Maximum number of consumers allowed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        a_invMax = 3000
        a_tMax = 1500
        a_outputDirectory = os.path.join(BASE_DIR, "output")

        # Ensure output directory exists
        Path(a_outputDirectory).mkdir(parents=True, exist_ok=True)

        # Initialize Topography
        topo = Topography(
            no_nodes=2,
            lattice_height=2,
            lattice_width=1,
            phi=0.3,
            envVar=1,
            skVec=np.array([0.1]),
            var_e=0.5,
            randGraph=False,
            gabriel=True,
            T_int=25.0,
            network_file="",
            scVec=np.array([0.05]),
            sc_file="",  # file for scaling 
            consArea_bin=np.array([1 if i % 2 == 0 else 0 for i in range(2)]),  #  binary conservation area
            consArea_multiplicative=True  # Conservation area perturbation mode
        )
        
        # Explicitly generate the network
        topo.gen_network()

        # Initialize Species
        spp = Species(
            topo=topo,
            c1=0.2,
            c2=0.25,
            c3=0.05,
            emRate=0.1,
            dispL=0.3,
            pProducer=0.4,
            prodComp=True,
            symComp=True,
            alpha=0.2,
            sigma=1.5,
            sigma_t=0.5,
            rho=0.3,
            comp_dist=1,
            omega=0.4,
            dispNorm=1.0,
            bodymass=1e-4,  # Ensure bodymass 
            mu=0.2          # Mortality rate
        )

        # Generate dispersal matrix
        spp.gen_disp_mat()
        print("Dispersal matrix generated successfully.")

        # Add a producer and a consumer
        spp.invade(0)  # Invade with a producer
        print("Producer added successfully.")
        spp.invade(1)  # Invade with a consumer
        print("Consumer added successfully.")

        # Initialize logB after initial invasions
        if spp.xMat.size > 0:
            spp.logB = np.log(spp.xMat + 1e-10)  # Shape: (species, nodes)
            logger.info("Initialized logB successfully.")
        else:
            logger.error("No species present after initial invasions.")
            raise ValueError("No species present after initial invasions.")


        # Set random seed for reproducibility
        g_seed = set_random_seed(FIX_SEED)
        print(f"Random Seed Set: {g_seed}")

        # Initialize Metacommunity
        meta = Metacommunity(
            spp=spp,
            a_init=a_init,
            a_bMat="",
            a_xMat="",
            a_scMat="",
            a_invMax=a_invMax,
            a_tMax=a_tMax,
            a_outputDirectory=a_outputDirectory,
            a_c1=0.2,
            a_c2=0.15,
            a_c3=0.05,
            a_emRate=0.1,
            a_dispL=0.3,
            a_pProducer=0.4,
            a_prodComp=True,
            a_symComp=True,
            a_alpha=0.2,
            a_sigma=1.5,
            a_sigma_t=0.5,
            a_rho=0.1,
            a_comp_dist=1,
            a_omega=0.6,
            a_dispNorm=1.0,
            a_no_nodes=2,
            a_lattice_height=2,
            a_lattice_width=1,
            a_phi=0.3,
            a_envVar=1,
            a_skVec=np.array([0.1]),
            a_var_e=0.5,
            a_randGraph=False,
            a_gabriel=True,
            a_T_int=25.0,
            a_envMat="",
            sc_file="",
            a_parOut=1.0,
            a_experiment="",
            a_rep=1,
            storeTraj=1,
            bMatFileName="",
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
        trajectory_PSD_states = []
        trajectory_PoissonClocks = []
        trajectory_logB = []               # To store logB per iteration
        trajectory_establishment_prob = []  # To store establishment probabilities
        trajectory_i = []                   # To store invasion rates

        max_species = meta.spp.xMat.shape[0]  # Initialize with initial number of species

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"Starting iteration {iteration} of assembly...")

            # Check for maximum species limit
            current_species = meta.spp.S_p + meta.spp.S_c  # Calculate total species
            current_producers = meta.spp.S_p
            current_consumers = meta.spp.S_c

            # Update max_species 
            if current_species > max_species:
                max_species = current_species

            # Record current state before invasions
            trajectory_iterations.append(iteration)
            trajectory_producers.append(current_producers)
            trajectory_consumers.append(current_consumers)
            trajectory_xMat.append(meta.spp.xMat.copy())  # xMat holds species abundances per node
            trajectory_rMat.append(meta.spp.rMat.copy())  # rMat holds species growth rates per node
            trajectory_PSD_states.append(meta.spp.waiting.copy())
            trajectory_PoissonClocks.append(meta.spp.poisson_clock.copy())
            trajectory_logB.append(meta.spp.logB.copy())
            trajectory_establishment_prob.append(meta.spp.establishment_prob.copy())
            trajectory_i.append(meta.spp.i.copy())

            # all species have gone extinct
            if current_species == 0:
                logger.info("All species have gone extinct. Stopping assembly.")
                break

            # Check for maximum producers
            if current_producers >= MAX_PRODUCERS:
                logger.info(f"Maximum producers limit of {MAX_PRODUCERS} reached.")
                producers_to_invade = 0
            else:
                producers_to_invade = INVADERS_PER_ITERATION.get(0, 0)
                # don't exceed MAX_SPECIES
                producers_to_invade = min(producers_to_invade, MAX_SPECIES - current_species)

            # Check for maximum consumers
            if current_consumers >= MAX_CONSUMERS:
                logger.info(f"Maximum consumers limit of {MAX_CONSUMERS} reached.")
                consumers_to_invade = 0
            else:
                consumers_to_invade = INVADERS_PER_ITERATION.get(1, 0)
                # don't exceed MAX_SPECIES
                consumers_to_invade = min(consumers_to_invade, MAX_SPECIES - current_species)

            # Invade producers
            if producers_to_invade > 0:
                logger.info(f"Invading {producers_to_invade} producer(s).")
                for _ in range(producers_to_invade):
                    meta.spp.invade(0)  # Call spp.invade with trophLev=0
                    meta.spp.I_p += 1  # Increment producer invasion count
            else:
                logger.info("No producers invaded this iteration.")

            # Invade consumers 
            if consumers_to_invade > 0:
                logger.info(f"Invading {consumers_to_invade} consumer(s).")
                for _ in range(consumers_to_invade):
                    meta.spp.invade(1)  # Call spp.invade with trophLev=1
                    meta.spp.I_c += 1  # Increment consumer invasion count
            else:
                logger.info("No consumers invaded this iteration.")

            # Simulate Dynamics with PSD Integration
            logger.info("Simulating dynamics with PSD integration...")
            meta.spp.simulate_psd(tmax=meta.tMax, stepsize=1, recording_stepsize=100)  # Adjust parameters 

            # Extinction Step
            logger.info("Removing extinct species...")
            meta.spp.extinct()

            # Update max_species 
            current_species = meta.spp.S_p + meta.spp.S_c
            if current_species > max_species:
                max_species = current_species

            # Stop condition: All invasions complete or all species extinct
            total_invasions = meta.spp.I_p + meta.spp.I_c
            if total_invasions >= meta.invMax:
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

        # Function to pad arrays
        def pad_array(arr, target_length, pad_value):
            current_length = arr.shape[0]
            if current_length < target_length:
                if arr.ndim == 1:
                    padding = np.full(target_length - current_length, pad_value, dtype=arr.dtype)
                    return np.concatenate([arr, padding])
                elif arr.ndim == 2:
                    num_cols = arr.shape[1]
                    padding = np.full((target_length - current_length, num_cols), pad_value, dtype=arr.dtype)
                    return np.vstack([arr, padding])
                else:
                    raise ValueError(f"pad_array only supports 1D or 2D arrays, but got {arr.ndim}D array.")
            elif current_length > target_length:
                if arr.ndim == 1:
                    return arr[:target_length]
                elif arr.ndim == 2:
                    return arr[:target_length, :]
                else:
                    raise ValueError(f"pad_array only supports 1D or 2D arrays, but got {arr.ndim}D array.")
            else:
                return arr

        # Debugging: Print lengths before padding
        print(f"Number of trajectory_producers: {len(trajectory_producers)}")
        print(f"Number of trajectory_consumers: {len(trajectory_consumers)}")
        print(f"Number of trajectory_iterations: {len(trajectory_iterations)}")
        print(f"Number of trajectory_xMat: {len(trajectory_xMat)}")
        print(f"Number of trajectory_rMat: {len(trajectory_rMat)}")
        print(f"Number of trajectory_PSD_states: {len(trajectory_PSD_states)}")
        print(f"Number of trajectory_PoissonClocks: {len(trajectory_PoissonClocks)}")
        print(f"Number of trajectory_logB: {len(trajectory_logB)}")
        print(f"Number of trajectory_establishment_prob: {len(trajectory_establishment_prob)}")
        print(f"Number of trajectory_i: {len(trajectory_i)}")
        print(f"Max species observed: {max_species}")

        # Verify that all trajectory lists have the same length
        expected_length = len(trajectory_producers)
        if not all(len(lst) == expected_length for lst in [
            trajectory_consumers,
            trajectory_iterations,
            trajectory_xMat,
            trajectory_rMat,
            trajectory_PSD_states,
            trajectory_PoissonClocks,
            trajectory_logB,
            trajectory_establishment_prob,
            trajectory_i
        ]):
            print("Error: Trajectory lists have inconsistent lengths.")
            print(f"Expected length: {expected_length}")
            print(f"Lengths: Producers: {len(trajectory_producers)}, Consumers: {len(trajectory_consumers)}, "
                  f"Iterations: {len(trajectory_iterations)}, xMat: {len(trajectory_xMat)}, "
                  f"rMat: {len(trajectory_rMat)}, PSD_states: {len(trajectory_PSD_states)}, "
                  f"PoissonClocks: {len(trajectory_PoissonClocks)}, logB: {len(trajectory_logB)}, "
                  f"Establishment_prob: {len(trajectory_establishment_prob)}, i: {len(trajectory_i)}")
            raise ValueError("Trajectory lists have inconsistent lengths. Cannot proceed with padding.")

        # Pad xMat and rMat trajectories
        trajectory_xMat_padded = []
        trajectory_rMat_padded = []
        for i in range(len(trajectory_xMat)):
            current_xMat = trajectory_xMat[i]
            current_num_species, num_nodes = current_xMat.shape
            if current_num_species < max_species:
                padding = np.zeros((max_species - current_num_species, num_nodes))
                padded_xMat = np.vstack((current_xMat, padding))
            elif current_num_species > max_species:
                padded_xMat = current_xMat[:max_species, :]
            else:
                padded_xMat = current_xMat
            trajectory_xMat_padded.append(padded_xMat)

            # Similarly for rMat
            current_rMat = trajectory_rMat[i]
            current_num_species, num_nodes = current_rMat.shape
            if current_num_species < max_species:
                padding = np.zeros((max_species - current_num_species, num_nodes))
                padded_rMat = np.vstack((current_rMat, padding))
            elif current_num_species > max_species:
                padded_rMat = current_rMat[:max_species, :]
            else:
                padded_rMat = current_rMat
            trajectory_rMat_padded.append(padded_rMat)

        # Pad PSD-related arrays
        trajectory_PSD_states_padded = []
        trajectory_PoissonClocks_padded = []
        trajectory_logB_padded = []
        trajectory_establishment_prob_padded = []
        trajectory_i_padded = []
        for i in range(len(trajectory_PSD_states)):
            # Pad 'waiting' with False
            padded_PSD_state = pad_array(trajectory_PSD_states[i], max_species, False)
            trajectory_PSD_states_padded.append(padded_PSD_state)

            # Pad 'poisson_clock' with 0.0
            padded_PoissonClock = pad_array(trajectory_PoissonClocks[i], max_species, 0.0)
            trajectory_PoissonClocks_padded.append(padded_PoissonClock)

            # Pad 'logB' with log(thresh) or a minimal value
            thresh = meta.spp.thresh if hasattr(meta.spp, 'thresh') else 1e-4
            logB_min = np.log(thresh + 1e-10)
            padded_logB = pad_array(trajectory_logB[i], max_species, logB_min)
            trajectory_logB_padded.append(padded_logB)

            # Pad 'establishment_prob' with 0.0
            padded_establishment_prob = pad_array(trajectory_establishment_prob[i], max_species, 0.0)
            trajectory_establishment_prob_padded.append(padded_establishment_prob)

            # Pad 'i' with 0.0
            padded_i = pad_array(trajectory_i[i], max_species, 0.0)
            trajectory_i_padded.append(padded_i)

        # Convert the padded lists to NumPy arrays
        trajectory_xMat = np.array(trajectory_xMat_padded)               # Shape: (num_records, max_species, nodes)
        trajectory_rMat = np.array(trajectory_rMat_padded)               # Shape: (num_records, max_species, nodes)
        trajectory_PSD_states = np.array(trajectory_PSD_states_padded)   # Shape: (num_records, max_species)
        trajectory_PoissonClocks = np.array(trajectory_PoissonClocks_padded)  # Shape: (num_records, max_species)
        trajectory_logB = np.array(trajectory_logB_padded)               # Shape: (num_records, max_species)
        trajectory_establishment_prob = np.array(trajectory_establishment_prob_padded)  # Shape: (num_records, max_species)
        trajectory_i = np.array(trajectory_i_padded)                     # Shape: (num_records, max_species)

        # Convert producers and consumers to NumPy arrays
        trajectory_producers = np.array(trajectory_producers)
        trajectory_consumers = np.array(trajectory_consumers)
        trajectory_iterations = np.array(trajectory_iterations)

        # Save trajectory data to a file for visualization
        trajectory_data = {
            'iteration': trajectory_iterations,
            'producers': trajectory_producers,
            'consumers': trajectory_consumers,
            'xMat': trajectory_xMat,                      # Shape: (num_records, max_species, nodes)
            'rMat': trajectory_rMat,                      # Shape: (num_records, max_species, nodes)
            'PSD_states': trajectory_PSD_states,          # Shape: (num_records, max_species)
            'PoissonClocks': trajectory_PoissonClocks,    # Shape: (num_records, max_species)
            'logB': trajectory_logB,                      # Shape: (num_records, max_species)
            'establishment_prob': trajectory_establishment_prob,  # Shape: (num_records, max_species)
            'i': trajectory_i,                             # Shape: (num_records, max_species)
        }
        # output directory 
        os.makedirs(meta.outputDirectory, exist_ok=True)
        trajectory_file = os.path.join(meta.outputDirectory, 'trajectory_data_PSD.npz')
        np.savez(trajectory_file, **trajectory_data)
        logger.info(f"Trajectory data saved to {trajectory_file}")

        # consistency checks after assembly
        meta.spp.consistency_checks()

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

# meta_c_dynamics method to use Assimulo
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
            # Assuming `compute_rhs` is properly implemented in CommunityDynamics
            dXdt = self.community_dynamics.compute_rhs(t, y, self.community_dynamics)
            return dXdt.flatten()

        problem = Explicit_Problem(rhs, y0, 0.0)
        solver = CVode(problem)
        # Set solver options 
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.maxsteps = 10000

        # Integrate until t_end
        solver.simulate(t_end)

        # Update state at the end of integration
        self.spp.xMat = solver.y.reshape(self.spp.xMat.shape)
        logger.debug(f"Dynamics simulated up to time {t_end}.")
        
        # **Compute and Update rMat Based on the New xMat**
        self.spp.rMat = self.community_dynamics.compute_intrinsic_growth_rates(self.spp.xMat)
        logger.debug(f"rMat updated based on new xMat.")
        
    except Exception as e:
        logger.error("An error occurred during dynamics simulation:")
        logger.error(traceback.format_exc())

# Attach the meta_c_dynamics method to Metacommunity class
if not hasattr(Metacommunity, 'meta_c_dynamics'):
    setattr(Metacommunity, 'meta_c_dynamics', meta_c_dynamics)

if __name__ == "__main__":
    # Profile the main function and save the profile to 'simulation_profile.prof'
    cProfile.run('main()', 'simulation_profile.prof')
