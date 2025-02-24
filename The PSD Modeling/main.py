############################################
# main.py
############################################
"""
Main driver script: sets up parameters, runs all four models (IBM, PSD, PSD2, ODE),
performs basic analysis, and saves data for advanced usage.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import numpy as np

# Import configuration parameters and model functions
from config import (
    TARGET_RICHNESS, BODY_MASS, INV, MORTALITY_RATE,
    RANDOM_SEED, TMAX, STEP_SIZE, N_RECORDS, RECORDING_STEP_SIZE,
    CONNECTANCE, INTERACTION_STRENGTH
)
from utils import setup_logging as setup_utils_logging
from models_ibm import IBMModel
from models_psd import PSDModel
from models_psd2 import PSD2Model
from models_ode import ODEModel
from analysis import (
    alpha_diversity,
    turnover_rate,
    plot_trajectories,
    plot_total_biomasses,
    histogram_biomass,
    covariance_matrix_plot,
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
    #### Axel will look at PSD2 later, skipping this for now
    logger.info("Axel will look at PSD2 later, skipping this for now")
    psd2_model = PSD2Model(r=r, C=C, tmax=RECORDING_STEP_SIZE, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    # psd2_model = PSD2Model(r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    (psd2_times, psd2_trajectory, psd2_waiting,
     psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob) = psd2_model.run()

    # RUN ODE Model
    logger.info("Running ODE Model...")
    ode_model = ODEModel(r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED)
    ode_times, ode_trajectory = ode_model.run()

    # Basic analysis and plotting
    logger.info("Generating basic analysis plots...")

    plot_trajectories(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Population Dynamics",
        filename_base="Trajectory.png"
    )

    plot_total_biomasses(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Population Dynamics: Total Biomass Comparison",
        filename="All_Models_Trajectory.png"
    )

    histogram_biomass(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        filename="Biomass_Distribution_Histogram.png"
    )

    ibm_turnover = turnover_rate(ibm_trajectory, RECORDING_STEP_SIZE)
    psd_turnover = turnover_rate(psd_trajectory, RECORDING_STEP_SIZE)
    psd2_turnover = turnover_rate(psd2_trajectory, RECORDING_STEP_SIZE)
    ode_turnover = turnover_rate(ode_trajectory, RECORDING_STEP_SIZE)

    ibm_alpha = alpha_diversity(ibm_trajectory)
    psd_alpha = alpha_diversity(psd_trajectory)
    psd2_alpha = alpha_diversity(psd2_trajectory)
    ode_alpha = alpha_diversity(ode_trajectory)

    logger.info("Alpha diversity medians:")
    logger.info(f"IBM: {np.median(ibm_alpha)}, PSD: {np.median(psd_alpha)}, PSD2: {np.median(psd2_alpha)}, ODE: {np.median(ode_alpha)}")
    logger.info("Mean+SE alpha (blockwise average, n=50) for IBM:")
    logger.info(mean_se(psd_alpha))

    covariance_matrix_plot(ibm_trajectory, "IBM Cov Matrix", "IBM_CovMatrix.png")
    covariance_matrix_plot(psd_trajectory, "PSD Cov Matrix", "PSD_CovMatrix.png")
    covariance_matrix_plot(psd2_trajectory, "PSD2 Cov Matrix", "PSD2_CovMatrix.png")
    covariance_matrix_plot(ode_trajectory, "ODE Cov Matrix", "ODE_CovMatrix.png")

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
