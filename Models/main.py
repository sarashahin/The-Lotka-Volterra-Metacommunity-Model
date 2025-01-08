

# import os
# import numpy as np
# import random
# import time
# import sys
# import scipy.io as sio
# from pathlib import Path
# from scipy.integrate import solve_ivp
# from datetime import datetime
# import signal
# import traceback
# import logging

# from metacommunity import Metacommunity  # Assuming the Metacommunity class and related classes are already implemented.
# from lvmcm_rng import LVMCM_rng
# from topography import Topography
# from species import Species


# # Configure logging for debugging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Constants for configuration
# FIX_SEED = 42
# ASSEMBLE = True
# OUTPUT = True
# SNAPSHOT = True
# SOURCE_SINK = False
# CMAT_REG = False
# TRAJECTORY = False
# FLUCTUATE = False
# WARMING = False
# LONGDISTDISP = False
# NODE_REMOVAL = False


# # Set random seed for reproducibility
# def set_random_seed(fixed_seed=0):
#     if fixed_seed > 0:
#         g_seed = 1
#     else:
#         g_seed = random.randint(1, 1000000)
    
#     # Create an instance of LVMCM_rng
#     rng_instance = LVMCM_rng()
#     rng_instance.set_seed(g_seed)  # Use set_seed to initialize with a specific seed
    
#     return g_seed

# def main():
#     try:
#         # Start the timer
#         start_time = time.time()
#         print("Starting metacommunity simulation...")

#         # Initialize simulation parameters
#         a_init = 1
#         a_invMax = 10
#         a_tMax = 500
#         a_outputDirectory = "output/"

#         # Ensure the output directory exists
#         Path(a_outputDirectory).mkdir(parents=True, exist_ok=True)

#         # Initialize Topography
#         topo = Topography(
#             no_nodes=32,
#             lattice_height=5,
#             lattice_width=5,
#             phi=0.3,
#             envVar=1,
#             skVec=np.array([0.1]),
#             var_e=0.5,
#             randGraph=False,
#             gabriel=True,
#             T_int=25.0,
#             network_file="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
#             scVec=np.array([0.05])  # Adjust if necessary
#         )
        
#         # Explicitly generate the network
#         topo.gen_network()

#         # Initialize Species
#         spp = Species(
#             topo=topo,
#             c1=0.5,
#             c2=0.3,
#             c3=0.1,
#             emRate=0.2,
#             dispL=0.7,
#             pProducer=0.7,
#             prodComp=True,
#             symComp=True,
#             alpha=0.02,
#             sigma=1.5,
#             sigma_t=0.3,
#             rho=0.8,
#             comp_dist=1,
#             omega=0.9,
#             dispNorm=1.0
#         )

#         # Generate dispersal matrix
#         spp.gen_disp_mat()
#         print("Dispersal matrix generated successfully.")

#         # Add a producer and a consumer
#         spp.invade(0)  # Invade with a producer
#         print("Producer added successfully.")
#         spp.invade(1)  # Invade with a consumer
#         print("Consumer added successfully.")

#         # Set random seed for reproducibility
#         g_seed = set_random_seed(FIX_SEED)
#         print(f"Random Seed Set: {g_seed}")

#         # Initialize Metacommunity
#         meta = Metacommunity(
#             spp=spp,
#             a_init=a_init,
#             a_bMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
#             a_xMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
#             a_scMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1params0.mat",
#             a_invMax=a_invMax,
#             a_tMax=a_tMax,
#             a_outputDirectory=a_outputDirectory,
#             a_c1=0.5,
#             a_c2=0.3,
#             a_c3=0.1,
#             a_emRate=0.2,
#             a_dispL=0.7,
#             a_pProducer=0.4,
#             a_prodComp=True,
#             a_symComp=True,
#             a_alpha=0.02,
#             a_sigma=1.5,
#             a_sigma_t=0.3,
#             a_rho=0.8,
#             a_comp_dist=1,
#             a_omega=0.9,
#             a_dispNorm=1.0,
#             a_no_nodes=32,
#             a_lattice_height=5,
#             a_lattice_width=5,
#             a_phi=0.3,
#             a_envVar=1,
#             a_skVec=np.array([0.1]),
#             a_var_e=0.5,
#             a_randGraph=False,
#             a_gabriel=True,
#             a_T_int=25.0,
#             a_envMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1rMat0.mat",
#             a_parOut=1.0,
#             a_experiment="/content/drive/MyDrive/Simulation_data/autonomous_turnover_example_pars.txt",
#             a_rep=1,
#             storeTraj=1,
#             bMatFileName="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
#             g_block_transitions=False
#         )

#         # Print parameters for verification
#         meta.print_params()

#         # Generate topography/environment
#         if a_init:
#             meta.spp.topo.gen_landscape()
#             print("Landscape generated successfully.")
#             meta.spp.gen_disp_mat()
#         else:
#             meta.spp.topo.gen_landscape(meta.spp.topo.network)
#             print("Landscape generated successfully.")

#         # Run the assembly process
#         if ASSEMBLE:
#             assemble_metacommunity(meta)

#         # Perform final bookkeeping
#         if OUTPUT:
#             final_bookkeeping(meta)

#         # End the simulation
#         elapsed_time = time.time() - start_time
#         print(f"Simulation completed successfully in {elapsed_time / 60:.2f} minutes.")

#     except Exception as e:
#         print("An error occurred during the simulation:")
#         traceback.print_exc()


# def assemble_metacommunity(meta):
#     """
#     Assemble the metacommunity by introducing invaders, simulating dynamics, 
#     and removing extinct species in each iteration.
#     """
#     try:
#         g_seed = set_random_seed(FIX_SEED)
#         print(f"Assembly process started with Seed: {g_seed}")

#         iteration = 0
#         while True:
#             iteration += 1
#             print(f"Starting iteration {iteration} of assembly...")

#             # Invader Testing
#             print("Invading producers and consumers...")
#             meta.invader_sample(0, 5)  # Invade with producers
#             meta.invader_sample(1, 5)  # Invade with consumers

#             # Simulate Dynamics
#             print("Simulating dynamics...")
#             meta.meta_c_dynamics(meta.tMax)

#             # Extinction Step
#             print("Removing extinct species...")
#             meta.spp.extinct()

#             # Stop condition: All invasions complete or all species extinct
#             if meta.spp.invasion >= meta.invMax:
#                 print("Maximum invasions reached. Stopping assembly.")
#                 break
#             if meta.spp.S_p == 0 and meta.spp.S_c == 0:
#                 print("All species extinct. Stopping assembly.")
#                 break

#             print(f"Iteration {iteration} complete.\n")

#     except Exception as e:
#         print("An error occurred during the assembly process:")
#         traceback.print_exc()


# def final_bookkeeping(meta):
#     """
#     Perform final bookkeeping steps after the simulation.
#     """
#     try:
#         print("Final bookkeeping...")
#         meta.meta_c_dynamics(1000)  # Final dynamics relaxation
#         meta.spp.extinct()  # Remove extinct species
#         meta.saveMC()
#         print("Bookkeeping complete.")
#     except Exception as e:
#         print("An error occurred during final bookkeeping:")
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()












import os
import numpy as np
import random
import time
import sys
import scipy.io as sio
from pathlib import Path
# Remove solve_ivp import
# from scipy.integrate import solve_ivp
from datetime import datetime
import signal
import traceback
import logging

# Import Assimulo classes for ODE solving
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from metacommunity import Metacommunity  # Assuming the Metacommunity class and related classes are already implemented.
from lvmcm_rng import LVMCM_rng
from topography import Topography
from species import Species

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
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
        a_init = 1
        a_invMax = 10
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
            network_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
            scVec=np.array([0.05])  # Adjust if necessary
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
            dispL=0.7,
            pProducer=0.7,
            prodComp=True,
            symComp=True,
            alpha=0.02,
            sigma=1.5,
            sigma_t=0.3,
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
            a_dispL=0.7,
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
            bMatFileName="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
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
        g_seed = set_random_seed(FIX_SEED)
        print(f"Assembly process started with Seed: {g_seed}")

        iteration = 0
        while True:
            iteration += 1
            print(f"Starting iteration {iteration} of assembly...")

            # Invader Testing
            print("Invading producers and consumers...")
            meta.invader_sample(0, 5)  # Invade producers
            meta.invader_sample(1, 5)  # Invade consumers

            # Simulate Dynamics with Assimulo
            print("Simulating dynamics...")
            meta.meta_c_dynamics(meta.tMax)

            # Extinction Step
            print("Removing extinct species...")
            meta.spp.extinct()

            # Stop condition: All invasions complete or all species extinct
            if meta.spp.invasion >= meta.invMax:
                print("Maximum invasions reached. Stopping assembly.")
                break
            if meta.spp.S_p == 0 and meta.spp.S_c == 0:
                print("All species extinct. Stopping assembly.")
                break

            print(f"Iteration {iteration} complete.\n")

    except Exception as e:
        print("An error occurred during the assembly process:")
        traceback.print_exc()


def final_bookkeeping(meta):
    """
    Perform final bookkeeping steps after the simulation.
    """
    try:
        print("Final bookkeeping...")
        meta.meta_c_dynamics(1000)  # Final dynamics relaxation
        meta.spp.extinct()  # Remove extinct species
        meta.saveMC()
        print("Bookkeeping complete.")
    except Exception as e:
        print("An error occurred during final bookkeeping:")
        traceback.print_exc()


# Example modification for meta_c_dynamics method to use Assimulo
# This should be placed in metacommunity.py or wherever meta_c_dynamics is defined.
# Here we show how to implement it inline for demonstration purposes.
def meta_c_dynamics(self, t_end):
    """
    Integrate the metacommunity dynamics up to time t_end using Assimulo.
    """
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


if __name__ == "__main__":
    main()
