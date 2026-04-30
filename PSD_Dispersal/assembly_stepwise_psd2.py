# ############################################
# # assembly_stepwise_psd2.py
# ############################################
# note: any comment with only new means it was added only for inifinit pool assembly

# -------- assembly_stepwise_psd2.py ------------------------------------
"""
Infinite-pool assembly for the PSD2 approximation – *vector* candidate mode
(1 + floor(frac*γ) fresh species tested in one PSD2 window).
"""
from __future__ import annotations
import numpy as np, logging, dispersal
from typing import Tuple
from models_psd2 import PSD2Model
from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS
from assembly_utils import draw_interactions, expand_RC, prune_extinct
# from config import ECOLOGICAL_MAX_B      # NEW

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

def stepwise_assembly_psd2(
        *,
        base_r        : float = 1.0,
        pressure_rate : float = 3e-3,  # per patch per time step
        # pressure_rate : float = 1e-2,
        window_time   : float = 5_000.,
        record_step   : float = 50.,
        F_sat         : int   = 12,      # saturation level (F_sat * γ)
        frac_multi    : float = 0.05,      # ← proportion of γ candidates/round
        max_rounds    : int   = 50,
        seed          : int   = 0,
        **model_kw
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    # ── founder ─────────────────────────────────────────────────────────
    r = np.array([base_r])
    C = np.array([[1.0]])
    # B = np.full((1, Ny, Nx), BODY_MASS/10)           # residents’ biomass (γ = 1)
    # ── founder starts in ONE random patch, just like IBM ──────────────
    B = np.zeros((1, Ny, Nx))
    iy0, ix0 = rng.integers(Ny), rng.integers(Nx)
    B[0, iy0, ix0] = BODY_MASS/10     # γ = 1
                   
    W  = np.zeros((1, Ny, Nx), dtype=bool)            # waiting flags
    PC = np.log(rng.random((1, Ny, Nx)))             # Poisson clocks
    attempts = 0

    gamma_file = open("/tmp/gamma.txt","w")
    
    for rnd in range(max_rounds):
        γ = len(r)
        n_cand = 1 + int(frac_multi * γ)

        # ── build enlarged (r,C) with ALL fresh candidates ──────────────
        r_big, C_big = r.copy(), C.copy()
        for _ in range(n_cand):
            row, col = draw_interactions(len(r_big), rng=rng)
            r_big, C_big = expand_RC(r_big, C_big, base_r, row, col)
            
        # ---- build initial biomass tensor ------------------------------
        # ---------- enlarge state for residents + n_cand --------------
        # B_big = np.zeros((len(r_big), Ny, Nx))
        # B_big[:γ] = B  
        
        B_big = np.zeros((len(r_big), Ny, Nx))
        B_big[:γ] = B                          # keep residents

        # seed each fresh candidate in ≤seed_size random patches
        seed_size = 10
        for k in range(n_cand):
            patches = rng.choice(Ny*Nx, min(seed_size, Ny*Nx), replace=False)
            B_big[γ+k].flat[patches] = BODY_MASS/10  
                                  # keep residents
        # candidates start at zero; propagule tap will feed them
         # New candidates arrive in D–state, so they start with *False*.
        W_big = np.ones_like(B_big, bool)              # new spp. start waiting
        W_big[:γ] = W  # keep residents old
        
        PC_big = np.empty_like(B_big)
        PC_big[:γ] = PC                                # keep resident clocks
        PC_big[γ:] = np.log(rng.random((n_cand, Ny, Nx)))


        # ── open propagule tap for EVERY candidate (last n_cand planes) ─
        # flux = np.zeros_like(B_big)
        # flux[-n_cand:] = pressure_rate * BODY_MASS
        # dispersal.set_invasion_pressure(flux)

        # propagule tap only in the very seed patches
        flux = np.zeros_like(B_big)
        for k in range(n_cand):
            patches = np.nonzero(B_big[γ+k].reshape(-1))[0]
            flux[γ+k].flat[patches] = pressure_rate * BODY_MASS

        dispersal.set_invasion_pressure(flux)

        # ── integrate ───────────────────────────────────────────────────
        model = PSD2Model(r_big, C_big,
                          initial_B=B_big,
                          initial_wait=W_big,
                          initial_clock=PC_big,
                          tmax=window_time, record_step=record_step,
                          dispersal_type='propagule', **model_kw)
        
        # _, B_traj, *_ = model.run()
        # final_B = B_traj[-1]
        
        _, B_traj, W_traj, PC_traj, *_ = model.run()
        final_B  = B_traj[-1]
        final_W  = W_traj[-1]
        final_PC = PC_traj[-1]

        dispersal.set_invasion_pressure(None)   # close tap
        attempts += n_cand

        # which of the *new* species invaded?
        estab_mask = final_B[-n_cand:].sum(axis=(1,2)) > 0
        if estab_mask.any():
            keep_new = np.where(estab_mask)[0] + γ
            keep_all = np.r_[np.arange(γ), keep_new]
            r, C  = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
            B, W, PC = final_B[keep_all], final_W[keep_all], final_PC[keep_all]
        else:
                # only residents survive
            B, W, PC = final_B[:γ], final_W[:γ], final_PC[:γ]

        # prune extinct residents (rare)
        alive = B.sum(axis=(1,2)) > BODY_MASS
        if (B.sum(axis=(1,2)) > 2*B.shape[1]*B.shape[2]).any():
            # B[B > 2] = 2
            # print("LARGE B DETECTED")
            log.warning(f"[PSD2] round {rnd}: large B detected")

            #sys.exit()
        if not alive.all():
            # r, C, *_ = prune_extinct(alive, r, C)
            r, C, B, keep_idx = prune_extinct(alive, r, C, B)
            W, PC = W[keep_idx], PC[keep_idx]

        γ = len(r)
        gamma_file.write(f"{rnd}, {attempts}, {γ}, {F_sat * γ}\n")
        gamma_file.flush()
        if attempts >= F_sat * γ:
            log.info(f"[PSD2] stop: attempts={attempts} ≥ {F_sat}×γ")
            break

    # -------- occupancy counts (needed for OFD) -------------------------
    # occ = (final_B[:len(r)] > 0).sum(axis=(1,2))   # patches per species
    # current biomass tensor B has exactly the surviving γ species
    occ = (B > 0).sum(axis=(1,2))                  # patches per species
    extra = dict(occ_counts=occ, B_seed=B.copy())
    gamma_file.close()
    return r, C, extra


# --------------------------------------------------------------------- #
# if __name__ == "__main__":
#     # simple smoke-test with default settings
#     r_fin, C_fin, extra = stepwise_assembly_psd2(window_time=400, record_step=20)
#     print("\nAssembled richness γ =", len(r_fin))
#     # print("final biomass shape:", B_fin.shape)



# stepwise_assembly_ibm returns four items
# (r, C, N, extra) – where N is the integer abundance tensor for the individual‑based model.

# # stepwise_assembly_psd2 returns three items because the PSD‑2 model
# # only needs the biomass tensor B to start with,
# # and that tensor is trivially derivable from r & C inside PSD2Model
# # if you choose to initialise with a generic value (it is done in the constructor when initial_B is omitted).