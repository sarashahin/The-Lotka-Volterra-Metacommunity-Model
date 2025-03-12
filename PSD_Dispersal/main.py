############################################
# main.py
############################################
"""
Main driver script.
Sets up simulation parameters, runs IBM, PSD, PSD2, and ODE models
(all multi-patch with dispersal and spatial heterogeneity), and performs basic analysis.
"""

import os
import sys
import logging
import numpy as np
import argparse
import config_utils
from logging.handlers import RotatingFileHandler
from utils import setup_logging, mean_se
import models_ibm as ibm_mod
import models_psd as psd_mod
import models_psd2 as psd2_mod
import models_ode as ode_mod
import analysis
from config import TARGET_RICHNESS, RANDOM_SEED, TMAX, RECORDING_STEP_SIZE, CONNECTANCE, INTERACTION_STRENGTH

############################################################
# Logging Setup
############################################################
def setup_logging():
    """
    Sets up logging so that:
      - All log messages at DEBUG level and above are recorded to a rotating file in the 'logs' folder.
      - Only INFO level and above are output to the console.
    Any existing handlers are cleared to avoid duplicate log messages.
    """
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

setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging is set up. Debug logs will be stored in the 'logs' folder.")

def main():
    logger.info("Setting up parameters and initial conditions...")

    # Parse configuration file argument.
    parser = argparse.ArgumentParser(description="Run spatial multi-patch simulation.")
    parser.add_argument("config_file", default="config.py", nargs='?', help="Path to configuration file.")
    args = parser.parse_args()
    config_module = config_utils.load_config(args.config_file)
    if config_module:
        logger.info(f"Read configuration file: {args.config_file}")
    else:
        sys.exit("Could not read config file.")
    config_utils.assign_all_config_variables(config_module, globals(), verbose=True)

    # Define number of species.
    S = TARGET_RICHNESS
    np.random.seed(RANDOM_SEED)

    # Create intrinsic growth rates and competition matrix.
    r = np.random.normal(loc=1.0, scale=0.1, size=S)
    adjacency = (np.random.rand(S, S) < CONNECTANCE).astype(float)
    C = INTERACTION_STRENGTH * adjacency
    np.fill_diagonal(C, 1.0)
    logger.info(f"Initial growth rates and competition matrix set for S={S} species.")

    # -------------------------
    # Run IBM multi-patch model
    # -------------------------
    logger.info("Running IBM multi-patch model...")
    ibm_trajectory, ibm_dispersal_flux = ibm_mod.IBMMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()

    # -------------------------
    # Run PSD multi-patch model
    # -------------------------
    logger.info("Running PSD multi-patch model...")
    (psd_log,
     psd_waiting,
     psd_growth,
     psd_invasion,
     psd_dispersal,
     psd_flux_traj) = psd_mod.PSDMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()

    psd_trajectory = np.exp(psd_log)  # convert log-biomass to biomass

    # --------------------------
    # Run PSD2 multi-patch model
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
    # Run ODE multi-patch model
    # -------------------------
    logger.info("Running ODE multi-patch model...")
    (ode_times,
     ode_trajectory,
     ode_growth,
     ode_invasion,
     ode_dispersal) = ode_mod.ODEMultiPatchModel(
        r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()

    # ---------------------------------------------------
    # Basic Analysis & Plotting (calling analysis.py funcs)
    # ---------------------------------------------------
    logger.info("Generating basic analysis plots...")

    # 1) Plot average trajectories
    analysis.plot_trajectories(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Population Dynamics",
        filename_base="Trajectory.png"
    )

    # 2) Plot total biomass
    analysis.plot_total_biomasses(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Population Dynamics: Total Biomass Comparison",
        filename="All_Models_Trajectory.png"
    )

    # 3) Histogram of biomass
    analysis.histogram_biomass(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        filename="Biomass_Distribution_Histogram.png"
    )

    # 4) Turnover rates
    ibm_turnover = analysis.turnover_rate(ibm_trajectory, RECORDING_STEP_SIZE)
    psd_turnover = analysis.turnover_rate(psd_trajectory, RECORDING_STEP_SIZE)
    psd2_turnover = analysis.turnover_rate(psd2_trajectory, RECORDING_STEP_SIZE)
    ode_turnover = analysis.turnover_rate(ode_trajectory, RECORDING_STEP_SIZE)

    # 5) Alpha diversity
    ibm_alpha = analysis.alpha_diversity(ibm_trajectory)
    psd_alpha = analysis.alpha_diversity(psd_trajectory)
    psd2_alpha = analysis.alpha_diversity(psd2_trajectory)
    ode_alpha = analysis.alpha_diversity(ode_trajectory)

    logger.info("Alpha diversity medians:")
    logger.info(f"IBM: {np.median(ibm_alpha)}, PSD: {np.median(psd_alpha)}, "
                f"PSD2: {np.median(psd2_alpha)}, ODE: {np.median(ode_alpha)}")

    logger.info("Mean+SE alpha (blockwise average, n=50) for PSD:")
    logger.info(mean_se(psd_alpha))

    # 6) Covariance matrix plot
    # analysis.covariance_matrix_plot(ibm_trajectory, "IBM Cov Matrix", "IBM_CovMatrix.png")
    # analysis.covariance_matrix_plot(psd_trajectory, "PSD Cov Matrix", "PSD_CovMatrix.png")
    # analysis.covariance_matrix_plot(psd2_trajectory, "PSD2 Cov Matrix", "PSD2_CovMatrix.png")
    # analysis.covariance_matrix_plot(ode_trajectory, "ODE Cov Matrix", "ODE_CovMatrix.png")

    # # 7) Another total biomass plot (could be redundant but safe)
    # analysis.plot_total_biomasses(
    #     trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
    #     labels=["IBM", "PSD", "PSD2", "ODE"],
    #     title="Total Biomass Across All Patches",
    #     filename="All_Models_Total_Biomass.png"
    # )

    # 8) Verify spatial variation for PSD2, species=10
    logger.info("=== Verifying PSD2 spatial variation for Species 10 ===")
    analysis.verify_spatial_variation(psd2_trajectory, species_index=10,
                                      plot_std=True, output_prefix="PSD2_spatial_check")

    # 9) Also verify flux variation for PSD2, species=20 (example)
    logger.info("=== Verifying PSD2 flux spatial variation for Species 20 ===")
    analysis.verify_spatial_variation(psd2_flux, species_index=20,
                                      plot_std=True, output_prefix="PSD2_flux_spatial_check")

    # 10) Plot some spatial heatmaps for PSD2
    for t_idx in [32, 33, 34]:
        analysis.plot_spatial_heatmap(psd2_trajectory,
                                      time_index=t_idx,
                                      species_index=10,
                                      title=f"PSD2: Species 10, Record {t_idx}",
                                      filename=f"psd2_species10_record{t_idx}.png")

    # 11) Spatial heatmap for IBM & ODE
    analysis.plot_spatial_heatmap(
        ibm_trajectory, time_index=50, species_index=20,
        title="IBM: Species 20, Record 50",
        filename="ibm_species20_record50.png"
    )
    analysis.plot_spatial_heatmap(
        ode_trajectory, time_index=50, species_index=30,
        title="ODE: Species 30, Record 50",
        filename="ode_species30_record50.png"
    )

    # 12) Visualize dispersal flux for IBM
    analysis.plot_dispersal_flux_heatmap(
        ibm_dispersal_flux,
        time_index=50,
        species_index=10,
        title="IBM Dispersal Flux at Record 50 (Species 10)",
        filename="ibm_dispersal_flux_record50.png"
    )
    analysis.animate_dispersal_flux(
        ibm_dispersal_flux,
        species_index=10,
        interval=200,
        save_filename="IBM_Dispersal_Flux_Animation.mp4"
    )

    # 13) Visualize dispersal flux for PSD2
    analysis.plot_dispersal_flux_heatmap(
        psd2_flux,
        time_index=50,
        species_index=20,
        title="PSD2 Dispersal Flux at Record 50 (Species 20)",
        filename="psd2_dispersal_flux_record50.png"
    )
    analysis.animate_dispersal_flux(
        psd2_flux,
        species_index=30,
        interval=200,
        save_filename="PSD2_Dispersal_Flux_Animation.mp4"
    )

    # 14) Animate spatial evolution for PSD2
    analysis.animate_spatial_evolution(
        psd2_trajectory, species_index=10, interval=200, save_filename="PSD2_Spatial_Animation.mp4"
    )

    # 15) Optional: visualize PSD dispersal flux at record=50, species=10
    analysis.plot_dispersal_flux_heatmap(
        psd_flux_traj,
        time_index=50,
        species_index=10,
        title="PSD Dispersal Flux at Record 50 (Species 10)",
        filename="psd_dispersal_flux_record50.png"
    )
    analysis.animate_dispersal_flux(
        psd_flux_traj,
        species_index=10,
        interval=200,
        save_filename="PSD_Dispersal_Flux_Animation.mp4"
    )

    # ------------------------------------------------
    # Save all model outputs including PSD2 flux
    # ------------------------------------------------
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
        psd2_flux,                      # PSD2 dispersal flux
        ode_trajectory
    )
    logger.info("Simulation completed. Outputs saved to 'model_outputs.npz'.")


if __name__ == "__main__":
    main()
