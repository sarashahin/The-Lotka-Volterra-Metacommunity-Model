############################################
# compare_engines.py
############################################
"""
A tool for statistically comparing the PSD2 and IBM simulation engines.

This script can load a community state from either an IBM or a PSD2
assembly checkpoint, then runs an ensemble of short simulations for both engines.
It reports statistical differences to quantify the divergence between the
deterministic approximation and the stochastic model.

Usage:
    # Compare from a PSD2 checkpoint
    python compare_engines.py /path/to/your/psd2_checkpoint.npz

    # Compare from an IBM checkpoint
    python compare_engines.py /path/to/your/ibm_checkpoint.npz
"""
import argparse
import logging
import numpy as np

# Project Module Imports
from config import BODY_MASS, STEP_SIZE, THRESHOLD
from models_psd2 import PSD2Model
from models_ibm import IBMModel
import simulation_utils as sim_utils

def main():
    """Main execution function for the comparison script."""
    parser = argparse.ArgumentParser(description="Statistically compare PSD2 and IBM engines from a checkpoint.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the assembly checkpoint .npz file (IBM or PSD2).")
    parser.add_argument("--n-replicates", type=int, default=10, help="Number of simulation replicates to run.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of the short simulation runs.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for the random number generator.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # --- 1. Load Initial State from Checkpoint ---
    logging.info(f"Loading community state from: {args.checkpoint_path}")
    is_psd2_checkpoint = False
    try:
        checkpoint = np.load(args.checkpoint_path, allow_pickle=True)
        r = checkpoint['r']
        C = checkpoint['C']
        
        # --- FIX: Load the detection threshold from the checkpoint, with a fallback. ---
        # The 'thr' key stores the biomass threshold used during assembly.
        thr_mass = checkpoint.get('thr', THRESHOLD) #
        logging.info(f"Using detection threshold (biomass): {thr_mass}")
        
        state_data = checkpoint['state']
        
        if state_data.ndim == 1:
            is_psd2_checkpoint = True
            logging.info("Detected PSD2 checkpoint (1D object array).")
            initial_psd2_state = tuple(state_data)
            initial_ibm_state_N = sim_utils.translate_psd_state_to_ibm(initial_psd2_state)
        
        elif state_data.ndim == 4:
            is_psd2_checkpoint = True
            logging.info("Detected PSD2 checkpoint (4D stacked array).")
            initial_psd2_state = (state_data[0], state_data[1], state_data[2])
            initial_ibm_state_N = sim_utils.translate_psd_state_to_ibm(initial_psd2_state)

        elif state_data.ndim == 3:
            logging.info("Detected IBM checkpoint (3D numerical array).")
            initial_ibm_state_N = state_data
            B, W, PC = sim_utils.translate_ibm_state_to_psd(initial_ibm_state_N)
            
            logging.info("Refining provisional PSD2 state based on local growth...")
            B_flat = B.reshape(len(r), -1)
            r_field = checkpoint.get('ENV_r_field', np.broadcast_to(r[:, np.newaxis, np.newaxis], B.shape))
            r_flat = r_field.reshape(len(r), -1)
            
            local_growth = r_flat - (C @ B_flat)
            local_growth = local_growth.reshape(B.shape)
            
            should_be_waiting = (initial_ibm_state_N == 0) & (local_growth > 0)
            
            W = should_be_waiting
            PC[~should_be_waiting] = 1.0
            
            initial_psd2_state = (B, W, PC)
        else:
            raise ValueError(f"Unrecognized state data shape in checkpoint: {state_data.shape}")

    except Exception as e:
        logging.error(f"Failed to load checkpoint file: {e}", exc_info=True)
        return

    master_rng = np.random.default_rng(args.seed)
    
    # --- 2. Run Simulation Ensemble ---
    psd2_final_states = []
    ibm_final_states = []

    logging.info(f"Starting ensemble of {args.n_replicates} replicates for a duration of {args.duration} time units.")

    for i in range(args.n_replicates):
        logging.info(f"--- Replicate {i+1}/{args.n_replicates} ---")

        replicate_seed = master_rng.integers(1e9)
        rng = np.random.default_rng(replicate_seed)

        # --- PSD2 Simulation ---
        logging.info("Running PSD2 simulation...")
        psd2_model = PSD2Model(r, C,
                               initial_B=initial_psd2_state[0],
                               initial_wait=initial_psd2_state[1],
                               initial_clock=initial_psd2_state[2],
                               tmax=args.duration, record_step=args.duration,
                               seed=replicate_seed)
        _, psd2_traj, *_ = psd2_model.run()
        psd2_final_states.append(psd2_traj[-1])

        # --- IBM Simulation ---
        logging.info("Running IBM simulation...")
        if is_psd2_checkpoint:
            N_for_this_replicate = sim_utils.translate_psd_state_to_ibm(initial_psd2_state, rng=rng)
        else:
            N_for_this_replicate = initial_ibm_state_N
            
        ibm_model = IBMModel(r, C,
                             initial_N=N_for_this_replicate,
                             nsteps=int(args.duration / STEP_SIZE),
                             record_step=int(args.duration / STEP_SIZE),
                             record_mode='full', seed=replicate_seed)
        ibm_traj = ibm_model.run()
        ibm_final_states.append(ibm_traj[-1])

    # --- 3. Statistical Analysis and Reporting ---
    logging.info("Ensemble finished. Analyzing results...")
    
    psd2_results = np.array(psd2_final_states)
    ibm_results = np.array(ibm_final_states)
    
    psd2_total_biomass = psd2_results.sum(axis=(1, 2, 3))
    ibm_total_biomass = ibm_results.sum(axis=(1, 2, 3))

    # --- FIX: Use the loaded thr_mass directly for richness calculation. ---
    psd2_gamma_richness = (psd2_results > thr_mass).any(axis=(2, 3)).sum(axis=1)
    ibm_gamma_richness = (ibm_results > thr_mass).any(axis=(2, 3)).sum(axis=1)

    print("\n" + "="*50)
    print("      ENGINE COMPARISON STATISTICAL REPORT")
    print("="*50)
    print(f"Based on {args.n_replicates} replicates from a single starting state.\n")

    print(f"Metric: Final Gamma Richness (γ)")
    print(f"  - PSD2: {np.mean(psd2_gamma_richness):.2f} ± {np.std(psd2_gamma_richness):.2f}")
    print(f"  - IBM:  {np.mean(ibm_gamma_richness):.2f} ± {np.std(ibm_gamma_richness):.2f}")
    print("-"*50)

    print(f"Metric: Final Total Biomass")
    print(f"  - PSD2: {np.mean(psd2_total_biomass):.4e} ± {np.std(psd2_total_biomass):.4e}")
    print(f"  - IBM:  {np.mean(ibm_total_biomass):.4e} ± {np.std(ibm_total_biomass):.4e}")
    print("="*50)

if __name__ == "__main__":
    main()
