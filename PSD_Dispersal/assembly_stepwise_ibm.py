############################################
# assembly_stepwise_ibm.py
############################################
# """
# Step‑wise community assembly.

# -------- assembly_stepwise_ibm.py -------------------------------------
"""
Infinite-pool assembly for the individual-based model (IBM) – tests
m = 1 + floor(frac*γ) candidates per window.
"""
from __future__ import annotations
import numpy as np, logging
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y
from assembly_utils import draw_interactions, expand_RC, prune_extinct

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

def stepwise_assembly_ibm(
        *,
        base_r        : float = 1.0,
        seed_size     : int   = 5,
        F_sat         : int   = 2,
        frac_multi    : float = 0.05,
        window_steps  : int   = 5_000,
        record_step   : int   = 50,
        max_rounds    : int   = 50,
        seed          : int   = 0,
        **model_kw):

    rng = np.random.default_rng(seed)

    # founder
    r = np.array([base_r])
    C = np.array([[1.0]])
    N = np.zeros((1, Ny, Nx), int)
    N[0, rng.integers(Ny), rng.integers(Nx)] = 1
    attempts = 0

    for rnd in range(max_rounds):
        γ = len(r)
        n_cand = 1 + int(frac_multi * γ)

        # ---- extend r,C,N with n_cand fresh species --------------------
        r_big, C_big = r.copy(), C.copy()
        N_big        = np.pad(N, [(0, n_cand), (0,0), (0,0)], constant_values=0)

        for add in range(n_cand):
            row, col = draw_interactions(len(r_big), rng=rng)
            r_big, C_big = expand_RC(r_big, C_big, base_r, row, col)

            # seed a few individuals
            patches = rng.choice(Ny*Nx, min(seed_size, Ny*Nx), replace=False)
            N_big[len(r)+add].flat[patches] = 1

        # ---- simulate ---------------------------------------------------
        model = IBMModel(r_big, C_big,
                         nsteps=window_steps, record_step=record_step,
                         initial_N=N_big, **model_kw)
        model.run()
        N_fin = model.N
        attempts += n_cand

        # keep only established newcomers
        estab_mask = N_fin[-n_cand:].sum(axis=(1,2)) > 0
        if estab_mask.any():
            keep_new = np.where(estab_mask)[0] + len(r)
            keep_all = np.r_[np.arange(len(r)), keep_new]
            r, C = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
            N    = N_fin[keep_all]
            log.info(f"[IBM] round {rnd}: {estab_mask.sum()}/{n_cand} "
                     f"candidates established  → γ={len(r)}")
        else:
            log.info(f"[IBM] round {rnd}: no candidates established")
            N = N_fin[:len(r)]  # drop all n_cand planes

        # prune extinct residents
        # alive = N.sum(axis=(1,2)) > 0
        alive = N.sum(axis=(1,2)) >= 3          # need ≥5 individuals in the metacommunity
        if not alive.all():
            r, C, N, _ = prune_extinct(alive, r, C, N)

        γ = len(r)
        if attempts >= F_sat * γ:
            log.info(f"[IBM] stop: attempts={attempts} ≥ {F_sat}×γ")
            break

    occ = (N > 0).sum(axis=(1,2))
    extra = dict(occ_counts=occ)
    return r, C, N, extra



# # ----------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     S = 5
#     r = np.ones(S)
#     C = np.eye(S) + 0.5*(np.ones((S,S)) - np.eye(S))
#     N = np.zeros((S, Ny, Nx), int)

#     final_residents = stepwise_assembly_ibm(
#         r, C, N,
#         window_steps=5_000,
#         record_step=200,
#         dispersal_type="propagule"
#     )
#     print("\nFinal community:", final_residents)
#     print("Gamma diversity:", len(final_residents))





