
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

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

def stepwise_assembly_psd2(
        *,
        base_r        : float = 1.0,
        pressure_rate : float = 1e-4,
        window_time   : float = 1_000.,
        record_step   : float = 50.,
        F_sat         : int   = 2,
        frac_multi    : float = 0.05,      # ← proportion of γ candidates/round
        max_rounds    : int   = 50,
        seed          : int   = 0,
        **model_kw
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    # ── founder ─────────────────────────────────────────────────────────
    r = np.array([base_r])
    C = np.array([[1.0]])
    attempts = 0

    for rnd in range(max_rounds):
        γ = len(r)
        n_cand = 1 + int(frac_multi * γ)

        # ── build enlarged (r,C) with ALL fresh candidates ──────────────
        r_big, C_big = r.copy(), C.copy()
        for _ in range(n_cand):
            row, col = draw_interactions(len(r_big), rng=rng)
            r_big, C_big = expand_RC(r_big, C_big, base_r, row, col)

        # ── open propagule tap for EVERY candidate (last n_cand planes) ─
        flux = np.zeros((len(r_big), Ny, Nx))
        flux[-n_cand:] = pressure_rate * BODY_MASS
        dispersal.set_invasion_pressure(flux)

        # ── integrate ───────────────────────────────────────────────────
        model = PSD2Model(r_big, C_big,
                          tmax=window_time, record_step=record_step,
                          dispersal_type='propagule', **model_kw)
        _, B_traj, *_ = model.run()
        final_B = B_traj[-1]

        dispersal.set_invasion_pressure(None)   # close tap
        attempts += n_cand

        # which of the *new* species invaded?
        estab_mask = final_B[-n_cand:].sum(axis=(1,2)) > 0
        if estab_mask.any():
            keep_new = np.where(estab_mask)[0] + len(r)  # indices in r_big
            keep_all = np.r_[np.arange(len(r)), keep_new]
            r, C = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
            log.info(f"[PSD2] round {rnd}: {estab_mask.sum()}/{n_cand} "
                     f"candidates established  → γ={len(r)}")
        else:
            log.info(f"[PSD2] round {rnd}: no candidates established")

        # prune extinct residents (rare)
        alive = final_B[:len(r)].sum(axis=(1,2)) > 0
        if not alive.all():
            r, C, *_ = prune_extinct(alive, r, C)

        γ = len(r)
        if attempts >= F_sat * γ:
            log.info(f"[PSD2] stop: attempts={attempts} ≥ {F_sat}×γ")
            break

    # -------- occupancy counts (needed for OFD) -------------------------
    occ = (final_B[:len(r)] > 0).sum(axis=(1,2))   # patches per species
    extra = dict(occ_counts=occ)




# ############################################
# # assembly_stepwise_psd2.py
# ############################################

# # -----------------------------------------------------------
# """Step-wise *infinite-pool* assembly for the PSD2 approximation.

# The routine keeps only resident + 1-candidate species in memory and
# uses a uniform, low invasion flux for the candidate (“propagule tap”).
# It stops once the heuristic attempts ≥ F_sat × γ is met.
# """

# from __future__ import annotations
# import numpy as np, logging, dispersal           # noqa: E401 (import order)
# from typing import Tuple
# from models_psd2 import PSD2Model
# from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS
# from assembly_utils import draw_interactions, expand_RC, prune_extinct

# log = logging.getLogger(__name__)
# Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X           # shorthand


# # --------------------------------------------------------------------- #
# def stepwise_assembly_psd2(
#     *,
#     base_r       : float = 1.0,          # intrinsic growth for every new sp.
#     pressure_rate: float = 1e-4,         # BODY_MASS · patch⁻¹ · time⁻¹
#     window_time  : float = 1000.0,
#     record_step  : float = 50.0,
#     F_sat        : int   = 2,            # stop when attempts ≥ F*γ
#     max_rounds   : int   = 50,
#     seed         : int   = 0,
#     **model_kw
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Returns
#     -------
#     r_final : (γ,)      intrinsic rates of the assembled community
#     C_final : (γ, γ)    competition matrix of the assembled community
#     """

#     rng = np.random.default_rng(seed)

#     # ── 0. initialise with ONE founder species ─────────────────────────
#     r = np.array([base_r], float)
#     C = np.array([[1.0]], float)
#     attempts = successes = 0

#     for rnd in range(max_rounds):
#         γ = len(r)                         # current richness

#         # ── 1. draw a *fresh* candidate from the infinite pool ────────
#         row, col = draw_interactions(γ, rng=rng)
#         r_new    = base_r                 # can be randomised 
#         r_cand, C_cand = expand_RC(r, C, r_new, row, col)

#         # ── 2. open propagule tap for *that* candidate only ───────────
#         flux = np.zeros((γ + 1, Ny, Nx))
#         flux[-1] = pressure_rate * BODY_MASS
#         dispersal.set_invasion_pressure(flux)

#         # ── 3. run PSD2 with residents + candidate ────────────────────
#         model = PSD2Model(r_cand, C_cand,
#                           tmax=window_time, record_step=record_step,
#                           dispersal_type="propagule",
#                           **model_kw)
#         _, B_traj, *_ = model.run()          # shape (records, S, Ny, Nx)
#         final_B = B_traj[-1]                 # biomass at t = window_time

#         # ── 4. close tap again ────────────────────────────────────────
#         dispersal.set_invasion_pressure(None)

#         attempts += 1
#         cand_established = final_B[-1].sum() > 0

#         if cand_established:
#             successes += 1
#             log.info(f"[assembly-PSD2] round {rnd}: species {γ} **established** "
#                      f"(γ → {γ+1})")
#             r, C = r_cand, C_cand                    # keep candidate
#         else:
#             log.info(f"[assembly-PSD2] round {rnd}: species {γ} failed")

#         # ── 5. prune any extinct *residents* (candidate handled above) ─
#         resident_alive = final_B[:-1].sum(axis=(1, 2)) > 0
#         if not resident_alive.all():
#             r, C, *_ = prune_extinct(resident_alive, r, C)

#         γ = len(r)                                     # updated richness

#         # ── 6. stopping rule (attempts ≥ F_sat × γ) ───────────────────
#         if attempts >= F_sat * γ:
#             log.info(f"[assembly-PSD2] saturation reached: attempts={attempts}, "
#                      f"γ={γ}")
#             break

#     return r, C


# # --------------------------------------------------------------------- #
# if __name__ == "__main__":
#     # simple smoke-test with default settings
#     r_fin, C_fin = stepwise_assembly_psd2(window_time=400, record_step=20)
#     print("\nAssembled richness γ =", len(r_fin))





    return r, C, extra

