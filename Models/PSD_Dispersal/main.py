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
from utils import setup_logging
import models_ibm as ibm_mod
import models_psd as psd_mod
import models_psd2 as psd2_mod
import models_ode as ode_mod
import analysis
from config import TARGET_RICHNESS, RANDOM_SEED, TMAX, RECORDING_STEP_SIZE, CONNECTANCE, INTERACTION_STRENGTH

# Setup logging.
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging is set up.")

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
    # >> NEW: The models will now use a per-patch r_field (inside each model) to introduce heterogeneity.
    r = np.random.normal(loc=1.0, scale=0.1, size=S)
    adjacency = (np.random.rand(S, S) < CONNECTANCE).astype(float)
    C = INTERACTION_STRENGTH * adjacency
    np.fill_diagonal(C, 1.0)
    logger.info(f"Initial growth rates and competition matrix set for S={S} species.")
    
    # Run IBM multi-patch model.
    logger.info("Running IBM multi-patch model...")
    # Note: The updated IBMMultiPatchModel now returns two outputs: biomass trajectory and its dispersal flux.
    ibm_trajectory, ibm_dispersal_flux = ibm_mod.IBMMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()
    
    # Run PSD multi-patch model.
    logger.info("Running PSD multi-patch model...")
    # Updated PSDMultiPatchModel now returns 6 outputs (including flux).
    psd_log, psd_waiting, psd_growth, psd_invasion, psd_dispersal, dispersal_flux_traj = psd_mod.PSDMultiPatchModel(
        r=r, C=C, nsteps=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()
    # Convert logB to biomass.
    psd_trajectory = np.exp(psd_log)
    
    # Run PSD2 multi-patch model.
    logger.info("Running PSD2 multi-patch model...")
    (psd2_times, psd2_trajectory, psd2_waiting,
     psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob, psd2_flux) = psd2_mod.PSD2MultiPatchModel(
        r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()
    
    # Run ODE multi-patch model.
    logger.info("Running ODE multi-patch model...")
    (ode_times, ode_trajectory, ode_growth, ode_invasion, ode_dispersal) = ode_mod.ODEMultiPatchModel(
        r=r, C=C, tmax=TMAX, record_step=RECORDING_STEP_SIZE, seed=RANDOM_SEED
    ).run()
    
    # Example analysis: plot total biomass across all patches.
    analysis.plot_total_biomasses(
        trajectories=[
            ibm_trajectory,
            psd_trajectory,
            psd2_trajectory,
            ode_trajectory
        ],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Total Biomass Across All Patches",
        filename="All_Models_Total_Biomass.png"
    )
    
    # 1) Plot average trajectories.
    analysis.plot_trajectories(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        title="Population Dynamics",
        filename_base="Trajectory.png"
    )
    
    # 3) Plot biomass histograms.
    analysis.histogram_biomass(
        trajectories=[ibm_trajectory, psd_trajectory, psd2_trajectory, ode_trajectory],
        labels=["IBM", "PSD", "PSD2", "ODE"],
        filename="Biomass_Distribution_Histogram.png"
    )
    
    # 8) Verify spatial variation for PSD2, species=10.
    logger.info("=== Verifying PSD2 spatial variation for Species 10 ===")
    analysis.verify_spatial_variation(psd2_trajectory, species_index=10,
                                      plot_std=True, output_prefix="PSD2_spatial_check")
    
    # 9) Also verify flux variation for PSD2, species=20.
    logger.info("=== Verifying PSD2 flux spatial variation for Species 20 ===")
    analysis.verify_spatial_variation(psd2_flux, species_index=20,
                                      plot_std=True, output_prefix="PSD2_flux_spatial_check")
    
    # Example verification: Plot heatmaps at several records.
    logger.info("=== 1) Checking multiple time points for PSD2, species 11 ===")
    for t_idx in [0, 10, 25, 50]:
        analysis.plot_spatial_heatmap(psd2_trajectory,
                                      time_index=t_idx,
                                      species_index=11,
                                      title=f"PSD2: species=11, record={t_idx}",
                                      filename=f"psd2_species11_record{t_idx}.png")
    
    logger.info("=== 2) Inspecting numeric values at record=50, species=0 ===")
    numeric_slice = psd2_trajectory[50, :, :, 0]
    logger.info(f"psd2_trajectory[50, :, :, 0]:\n{numeric_slice}")
    
    logger.info("=== 3) Adjusting parameters in config.py to reduce dispersal if needed ===")
    logger.info("For stronger spatial heterogeneity, consider lowering DISPERSAL_RATE or using a DISPERSAL_FIELD.")
    
    logger.info("=== 4) Visualizing a different species/time ===")
    analysis.plot_spatial_heatmap(psd2_trajectory,
                                  time_index=50,
                                  species_index=10,
                                  title="PSD2: species=10, record=50",
                                  filename="psd2_species10_record50.png")
    
    # 13) Visualize dispersal flux for PSD2.
    analysis.plot_dispersal_flux_heatmap(
        psd2_flux,
        time_index=50,
        species_index=20,
        title="PSD2 Dispersal Flux at Record 50 (Species 20)",
        filename="psd2_dispersal_flux_record50.png"
    )
    
    # Also, plot IBM and ODE spatial heatmaps.
    analysis.plot_spatial_heatmap(
        ibm_trajectory,
        time_index=50,
        species_index=10,
        title="IBM: species=10, record=50",
        filename="ibm_species10_record50.png"
    )
    analysis.plot_spatial_heatmap(
        ode_trajectory,
        time_index=50,
        species_index=3,
        title="ODE: species=3, record=50",
        filename="ode_species03_record50.png"
    )
    
    # Animate the spatial evolution for PSD2 biomass for species 10.
    analysis.animate_spatial_evolution(psd2_trajectory, species_index=10, interval=200,
                                       save_filename="PSD2_Spatial_Animation.mp4")
    
    # Verify spatial variation for PSD2, species 20.
    analysis.verify_spatial_variation(psd2_trajectory, species_index=20, plot_std=True, 
                                      output_prefix="PSD2_spatial_check")
    
    # ADDED: Visualize IBM dispersal flux.
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
    
    # ADDED: Visualize PSD dispersal flux.
    analysis.plot_dispersal_flux_heatmap(
        dispersal_flux_traj,
        time_index=50,
        species_index=10,
        title="PSD Dispersal Flux at Record 50 (Species 10)",
        filename="psd_dispersal_flux_record50.png"
    )
    analysis.animate_dispersal_flux(
        dispersal_flux_traj,
        species_index=10,
        interval=200,
        save_filename="PSD_Dispersal_Flux_Animation.mp4"
    )
    
    # ADDED: Save model outputs including dispersal flux arrays.
    analysis.save_model_output(
        ibm_trajectory,
        ibm_dispersal_flux,          # CHANGED: IBM flux output added
        psd_trajectory,
        dispersal_flux_traj,         # CHANGED: PSD flux output added
        psd2_trajectory,
        psd2_waiting,
        psd2_poisson_clock,
        psd2_growth_rate,
        psd2_invasion_rate,
        psd2_est_prob,
        psd2_flux,                   # CHANGED: PSD2 flux output added
        ode_trajectory
    )
    logger.info("Simulation completed. Outputs saved to 'model_outputs.npz'.")

if __name__ == "__main__":
    main()
