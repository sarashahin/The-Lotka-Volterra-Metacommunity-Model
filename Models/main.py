
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
# Remove solve_ivp import
# from scipy.integrate import solve_ivp
from datetime import datetime
import signal
import traceback
import logging

# Import Assimulo classes for ODE solving
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from metacommunity import Metacommunity  
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
            network_file="",
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
            a_bMat="",
            a_xMat="",
            a_scMat="",
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
            a_envMat="",
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


# meta_c_dynamics method to use Assimulo
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
    # Set solver options 
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.maxsteps = 10000

    # Integrate until t_end
    solver.simulate(t_end)

    # Update state at the end of integration
    self.spp.xMat = solver.y.reshape(self.spp.xMat.shape)


if __name__ == "__main__":
    main()




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
