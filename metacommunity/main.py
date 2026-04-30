############################################
# main.py
############################################
"""
Main driver script.
Sets up simulation parameters, runs IBM, PSD, PSD2, and ODE models
(all multipatch with dispersal and spatial heterogeneity), and performs basic analysis.
Outputs:
  - Essential plots are saved in the "plots" folder.
  - Model outputs are saved in the "outputs" folder.
  - A statistical report is printed, including total biomass and state transition metrics (e.g., waiting→active).
The research aim is to investigate real‐world spatiotemporal multicommunities (with three state transitions)
using classic and quantum‐inspired dispersal, with biodiversity mapping.
"""

import os
import sys
import logging
import numpy as np
import argparse
from logging.handlers import RotatingFileHandler

# Import configuration utilities and analysis functions.
import config_utils
from utils import setup_logging, mean_se

# Import model modules and analysis functions.
import models_ibm as ibm_mod
import models_psd as psd_mod
import models_psd2 as psd2_mod
import models_ode as ode_mod
import analysis

from config import TARGET_RICHNESS, RANDOM_SEED, TMAX, RECORDING_STEP_SIZE, CONNECTANCE, INTERACTION_STRENGTH, NUM_PATCHES_X, NUM_PATCHES_Y

############################################################
# Logging Setup
############################################################
def setup_logging_custom():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(
        os.path.join(log_folder, "debug.log"),
        maxBytes=100000,
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

setup_logging_custom()
logger = logging.getLogger(__name__)
logger.info("Logging is set up. Debug logs will be stored in the 'logs' folder.")

############################################################
# Create Output Directories
############################################################
PLOTS_DIR = "plots"
OUTPUTS_DIR = "outputs"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

############################################################
# Function: Print Model Statistics and State Transitions
############################################################
def print_model_statistics(ibm_traj, psd_traj, psd_wait, psd2_traj, psd2_wait, ode_traj):
    total_patches = NUM_PATCHES_X * NUM_PATCHES_Y * ibm_traj.shape[-1]
    
    # IBM Model (no waiting state in IBM dynamics)
    ibm_total = np.sum(ibm_traj, axis=(1,2))
    print("=== IBM Model Total Biomass ===")
    print("Mean:", np.mean(ibm_total),
          "Median:", np.median(ibm_total),
          "Std:", np.std(ibm_total))
    
    # PSD Model: waiting state statistics and transitions.
    psd_total = np.sum(psd_traj, axis=(1,2))
    waiting_counts_psd = np.sum(psd_wait, axis=(1,2,3))
    active_counts_psd = total_patches - waiting_counts_psd
    # Count transitions between waiting and active between consecutive records.
    transitions_psd = np.sum(np.abs(np.diff(psd_wait.astype(int), axis=0)), axis=(1,2,3))
    print("\n=== PSD Model ===")
    print("Total Biomass: Mean =", np.mean(psd_total),
          "Median =", np.median(psd_total),
          "Std =", np.std(psd_total))
    print("Waiting patches per record:", waiting_counts_psd)
    print("Active patches per record:", active_counts_psd)
    print("Transitions (waiting↔active) per record interval:", transitions_psd)
    
    # PSD2 Model: waiting state statistics and transitions.
    psd2_total = np.sum(psd2_traj, axis=(1,2))
    waiting_counts_psd2 = np.sum(psd2_wait, axis=(1,2,3))
    active_counts_psd2 = total_patches - waiting_counts_psd2
    transitions_psd2 = np.sum(np.abs(np.diff(psd2_wait.astype(int), axis=0)), axis=(1,2,3))
    print("\n=== PSD2 Model ===")
    print("Total Biomass: Mean =", np.mean(psd2_total),
          "Median =", np.median(psd2_total),
          "Std =", np.std(psd2_total))
    print("Waiting patches per record:", waiting_counts_psd2)
    print("Active patches per record:", active_counts_psd2)
    print("Transitions (waiting↔active) per record interval:", transitions_psd2)
    
    # ODE Model
    ode_total = np.sum(ode_traj, axis=(1,2))
    print("\n=== ODE Model Total Biomass ===")
    print("Mean:", np.mean(ode_total),
          "Median:", np.median(ode_total),
          "Std:", np.std(ode_total))


############################################################
# Main Function
############################################################
def main():
    logger.info("Setting up parameters and initial conditions...")

    # Parse command-line arguments for configuration file.
    parser = argparse.ArgumentParser(description="Run spatial multi-patch simulation.")
    parser.add_argument("config_file", default="config.py", nargs='?', help="Path to configuration file.")
    args = parser.parse_args()
    config_module = config_utils.load_config(args.config_file)
    if config_module:
        logger.info(f"Read configuration file: {args.config_file}")
    else:
        sys.exit("Could not read configuration file.")
    config_utils.assign_all_config_variables(config_module, globals(), verbose=True)
    modules_to_configure = [ibm_mod, psd_mod, psd2_mod, ode_mod, analysis]
    config_utils.configure_modules(config_module, modules_to_configure)

    # Define number of species from config.
    S = TARGET_RICHNESS
    np.random.seed(RANDOM_SEED)

    # Create intrinsic growth rates and competition matrix.
    r = np.random.normal(loc=1.0, scale=0.1, size=S)
    adjacency = (np.random.rand(S, S) < CONNECTANCE).astype(float)
    C = INTERACTION_STRENGTH * adjacency
    np.fill_diagonal(C, 1.0)
    logger.info(f"Initial growth rates and competition matrix set for S={S} species.")

    # -------------------------
    # Run IBM Multi-Patch Model
    # -------------------------
    logger.info("Running IBM multi-patch model...")
    ibm_trajectory, ibm_dispersal_flux = ibm_mod.IBMMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()

    # -------------------------
    # Run PSD Multi-Patch Model
    # -------------------------
    logger.info("Running PSD multi-patch model...")
    (psd_log,
     psd_waiting,
     psd_growth,
     psd_invasion,
     psd_dispersal,
     psd_flux_traj) = psd_mod.PSDMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run() # Convert logB to biomass

    psd_trajectory = np.exp(psd_log)  # convert log-biomass to biomass

    # --------------------------
    # Run PSD2 Multi-Patch Model
    # --------------------------
    logger.info("Running PSD2 multi-patch model...")
    # We capture the extra dispersal flux output (psd2_flux).
    (psd2_times,
     psd2_trajectory,
     psd2_waiting,
     psd2_poisson_clock,
     psd2_growth_rate,
     psd2_invasion_rate,
     psd2_est_prob,
     psd2_flux) = psd2_mod.PSD2MultiPatchModel(
        r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()
    # -------------------------
    # Run ODE Multi-Patch Model
    # -------------------------
    logger.info("Running ODE multi-patch model...")
    (ode_times,
     ode_trajectory,
     ode_growth,
     ode_invasion,
     ode_dispersal) = ode_mod.ODEMultiPatchModel(
        r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()

    ############################################################
    # Print Statistical Report
    ############################################################
    print("\n=== MODEL STATISTICS ===")
    print_model_statistics(ibm_trajectory, psd_trajectory, psd_waiting, psd2_trajectory, psd2_waiting, ode_trajectory)

    ############################################################
    # Analysis & Essential Plots
    ############################################################
    logger.info("Generating essential analysis plots...")

    # Total Biomass vs Time Plot (multi-trajectory)
    total_biomass_models = [ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory]

    model_labels = ["IBM", "PSD", "PSD2", "ODE"]
    biomass_plot_filename = os.path.join(PLOTS_DIR, "Total_Biomass_Comparison.png")
    analysis.plot_total_biomasses(
        trajectories=total_biomass_models,
        labels=model_labels,
        title="Total Biomass vs Time",
        filename=biomass_plot_filename
    )

    # Biomass Distribution Histogram (log10 scale)
    hist_filename = os.path.join(PLOTS_DIR, "Biomass_Distribution_Histogram.png")
    analysis.histogram_biomass(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=model_labels,
        filename=hist_filename
    )

    # Heatmaps for dispersal diagnostics (for IBM, PSD, ODE, and invasion flux for PSD2)
    def plot_heatmap(data, title, filename):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(title)
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved figure {filename}")

    rec_idx = -1  # Use last recording for heatmap
    ibm_disp = ibm_dispersal_flux[rec_idx, :, :, 0]
    plot_heatmap(ibm_disp, "IBM Dispersal Flux (Species 0)", os.path.join(PLOTS_DIR, "IBM_Dispersal_Heatmap.png"))

    
    psd_disp = psd_dispersal[rec_idx, :, :, 0]
    plot_heatmap(psd_disp, "PSD Dispersal Flux (Species 0)", os.path.join(PLOTS_DIR, "PSD_Dispersal_Heatmap.png"))
    
    ode_disp = ode_dispersal[rec_idx, :, :, 0]
    plot_heatmap(ode_disp, "ODE Dispersal Flux (Species 0)", os.path.join(PLOTS_DIR, "ODE_Dispersal_Heatmap.png"))
    
    psd2_disp = psd2_invasion_rate[rec_idx, :, :, 0]
    plot_heatmap(psd2_disp, "PSD2 Invasion Rate (Species 0)", os.path.join(PLOTS_DIR, "PSD2_Invasion_Heatmap.png"))

    ############################################################
    # Save Model Outputs for Advanced Analysis
    ############################################################
    output_filepath = os.path.join(OUTPUTS_DIR, "model_outputs.npz")
    analysis.save_model_output(
        ibm_trajectory,                 # IBM
        psd_trajectory,                 # PSD
        psd_flux_traj,                  # PSD flux
        psd2_trajectory,                # PSD2
        psd2_waiting,
        psd2_poisson_clock,
        psd2_growth_rate,
        psd2_invasion_rate,
        psd2_est_prob,
        psd2_flux,                      
        ode_trajectory,
        filename=output_filepath
    )
    logger.info(f"All done. Model outputs saved to '{output_filepath}' for advanced analysis.")

if __name__ == "__main__":
    main()
