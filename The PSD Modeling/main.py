############################################
# main.py
############################################
"""
Main driver script: sets up parameters, runs all four models (IBM, PSD, ODE),
performs basic analysis, and saves data for advanced usage.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

# To import configuration file:
import argparse
import config_utils

from utils import setup_logging as setup_utils_logging
import models_ibm
import models_psd
import models_psd2
import models_ode
import analysis
from models_ode import ODEModel
from models_ibm import IBMModel
from models_psd import PSDModel
from models_psd2 import PSD2Model

from analysis import (
    alpha_diversity,
    turnover_rate,
    plot_trajectories,
    # plot_total_biomasses,
    histogram_biomass,
    # covariance_matrix_plot,
    save_model_output,
    mav, bav, mean_se
)

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

############################################################
# Main Function
############################################################
def main():
    logger.info("Setting up parameters and initial conditions...")

    # Parse command-line arguments. Currently only for loading config file.
    parser = argparse.ArgumentParser(description="Run the script with a configuration file.")
    parser.add_argument("config_file",
                        default="config.py",
                        nargs='?', # fall back to default if not present
                        help="Path to the configuration .py file.")
    
    parser.add_argument("--run-time", type=float, default=None,
                        help="override TMAX (model units)")
    parser.add_argument("--bins", default="auto",
                        help="histogram bins: int | auto | auto:Scott | auto:FD")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for THIS run (overrides RANDOM_SEED ""in config.py)")
    
    parser.add_argument("--burn-in", type=float, default=0.2,
                    help="fraction of the trajectory to discard "
                         "before histogramming (0‒1, default 0.2)")

    args = parser.parse_args()
    
    # ------------------------------------------------------------------
    # 1 · load the config file  (this defines TMAX, RECORDING_STEP_SIZE, …)
    # ------------------------------------------------------------------
    config_path   = args.config_file
    config_module = config_utils.load_config(config_path)
    if not config_module:
        sys.exit(f"Could not read config file {config_path!r}")
    config_utils.assign_all_config_variables(config_module, globals(), verbose=True)

    # ------------------------------------------------------------------
    # 2 · optionally override seed and TMAX
    # ------------------------------------------------------------------
    if args.seed is not None:
        RANDOM_SEED = args.seed
        logger.info("Overriding RANDOM_SEED → %d", RANDOM_SEED)

    # ------------------------------------------------------------------
    # 3 · turn the --bins string into an object for histogram_biomass
    # ------------------------------------------------------------------
    if str(args.bins).isdigit():
        bin_setting = int(args.bins)
    elif str(args.bins).lower().startswith("auto"):
        rule = "FD" if ":" not in args.bins else args.bins.split(":")[1]
        bin_setting = ("auto", rule)          # e.g. ("auto","Scott")
    else:
        raise ValueError("--bins must be int or auto[:Rule]")

    # ------------------------------------------------------------------
    # 4 · configure helper modules that rely on globals from config.py
    # ------------------------------------------------------------------
    modules_to_configure = [models_ibm, models_psd, models_psd2, analysis]
    config_utils.configure_modules(config_module, modules_to_configure)

    # Define number of species (TARGET_RICHNESS)
    S = TARGET_RICHNESS

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Create intrinsic growth rates and competition matrix
    r = np.random.normal(loc=1.0, scale=0.1, size=S)
    adjacency = (np.random.rand(S, S) < CONNECTANCE).astype(float)
    C = INTERACTION_STRENGTH * adjacency
    np.fill_diagonal(C, 1.0)

    logger.info(f"Initial growth rates (r) and competition matrix (C) set for S={S} species.")
    
    # RUN ODE Model
    logger.info("Running ODE Model...")
    ode_model = ODEModel(r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    ode_times, ode_trajectory = ode_model.run()


    # RUN IBM Model
    logger.info("Running IBM Model...")
    ibm_model = IBMModel(r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    ibm_trajectory = ibm_model.run()

    # RUN PSD Model
    logger.info("Running PSD Model...")
    psd_model = PSDModel(r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    psd_trajectory_log, psd_waiting = psd_model.run()
    psd_trajectory = np.exp(psd_trajectory_log) * (~psd_waiting)

    # RUN PSD2 Model ( with extra diagnostic outputs)
    logger.info("Running PSD2 Model...")
    psd2_model = PSD2Model(r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    (psd2_times, psd2_trajectory, psd2_waiting,
     psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob) = psd2_model.run()

    # Basic analysis and plotting
    logger.info("Generating basic analysis plots...")

    plot_trajectories(
        trajectories=[ode_trajectory, ibm_trajectory, psd_trajectory],
        labels=["ODE", "IBM", "PSD"],
        title="Population Dynamics",
        filename_base="Trajectory.png"
    )

    # plot_total_biomasses(
    #     trajectories=[ode_trajectory, ibm_trajectory, psd_trajectory, psd2_trajectory],
    #     labels=["ODE", "IBM", "PSD", "PSD2"],
    #     title="Population Dynamics: Total Biomass Comparison",
    #     filename="All_Models_Trajectory.png"
    # )
    



    histogram_biomass(
        trajectories=[ode_trajectory, ibm_trajectory, psd_trajectory],
        labels=["ODE", "IBM", "PSD"],
        filename="Biomass_Distribution_Histogram.png",
        bins=40,
        burn_in_frac=args.burn_in
    )


    ibm_alpha = alpha_diversity(ibm_trajectory)
    psd_alpha = alpha_diversity(psd_trajectory)
    psd2_alpha = alpha_diversity(psd2_trajectory)

    logger.info("Alpha diversity medians:")
    logger.info(f"IBM: {np.median(ibm_alpha)}, PSD2: {np.median(psd2_alpha)}")
    logger.info("Mean+SE alpha (blockwise average, n=50) for IBM:")
    logger.info(mean_se(ibm_alpha))
    # logger.info("Mean+SE alpha (blockwise average, n=50) for PSD:")
    # logger.info(mean_se(psd_alpha))
    logger.info("Mean+SE alpha (blockwise average, n=50) for PSD2:")
    logger.info(mean_se(psd2_alpha))

    # Save outputs for advanced visualization including PSD2 diagnostics
    save_model_output(
        ibm_trajectory,
        psd_trajectory,
        psd2_trajectory,
        psd2_waiting,
        psd2_poisson_clock,
        psd2_growth_rate,
        psd2_invasion_rate,
        psd2_est_prob,
        ode_trajectory
    )
    logger.info("All done. Model outputs saved to 'model_outputs.npz' for advanced usage.")

if __name__ == "__main__":
    main()
