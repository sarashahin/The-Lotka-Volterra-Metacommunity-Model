
############################################
# assembly_stepwise_ibm.py
############################################
"""
Step‑wise community assembly.

* one candidate per round;
* seed with a few individuals;
* run, prune extinct species, keep survivors’
  abundances;
* stop when richness no longer rises.
"""
from __future__ import annotations
import numpy as np, logging
from models_ibm import IBMModel
from typing import Optional
from config import NUM_PATCHES_X, NUM_PATCHES_Y
from assembly_utils import draw_interactions, expand_RC, prune_extinct

log = logging.getLogger(__name__)

def stepwise_assembly_ibm(
        *,
        base_r        : float = 1.0,
        seed_size     : int   = 5,
        F_sat         : int   = 2,
        window_steps  : int   = 5_000,
        record_step   : int   = 50,
        max_rounds    : int   = 50,
        seed          : int   = 0,
        **model_kw):

    rng = np.random.default_rng(seed)
    Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

    # ---------- 0. founder ----------------------------------------------
    r   = np.array([base_r], float)
    C   = np.array([[1.0]], float)
    N   = np.zeros((1, Ny, Nx), int)
    N[0, rng.integers(Ny), rng.integers(Nx)] = 1

    attempts, successes = 0, 0

    for rnd in range(max_rounds):

        # ---------- 1. fresh candidate ----------------------------------
        row, col = draw_interactions(len(r), rng=rng)
        r_new    = base_r
        r_cand, C_cand = expand_RC(r, C, r_new, row, col)

        # expand counts with zeros + seeding
        N_cand = np.pad(N, [(0,1),(0,0),(0,0)], constant_values=0)
        
        # clamp seed_size to the number of patches
        max_seeds = Ny * Nx
        n_seeds   = min(seed_size, max_seeds)
        pick = rng.choice(max_seeds, n_seeds, replace=False)
        # pick = rng.choice(Ny*Nx, seed_size, replace=False)
        N_cand[-1].flat[pick] = 1

        # ---------- 2. run IBM on residents+candidate -------------------
        model = IBMModel(r_cand, C_cand,
                         nsteps=window_steps, record_step=record_step,
                         initial_N=N_cand, **model_kw)
        model.run()
        N_final = model.N                                   # (S,Ny,Nx)

        attempts += 1
        established = N_final[-1].sum() > 0
        if established:
            successes += 1
            log.info(f"[IBM]  sp {len(r)} established (γ→{len(r)+1})")
            r, C, N = r_cand, C_cand, N_final
        else:
            log.info(f"[IBM]  sp {len(r)} failed")
            N = N_final[:-1]                                # drop candidate plane

        # ---------- 3. prune extinct residents --------------------------
        mask_alive = N.sum(axis=(1,2)) > 0
        if not mask_alive.all():
            r, C, N, _ = prune_extinct(mask_alive, r, C, N)

        gamma = len(r)
        if attempts >= F_sat * gamma:
            log.info(f"[IBM]  saturation heuristic : attempts={attempts}, γ={gamma}")
            break

    return r, C, N



# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    S = 5
    r = np.ones(S)
    C = np.eye(S) + 0.5*(np.ones((S,S)) - np.eye(S))

    final_residents = stepwise_assembly(
        r, C,
        window_steps=5_000,
        record_step=200,
        dispersal_type="propagule"
    )
    print("\nFinal community:", final_residents)
    print("Gamma diversity:", len(final_residents))

