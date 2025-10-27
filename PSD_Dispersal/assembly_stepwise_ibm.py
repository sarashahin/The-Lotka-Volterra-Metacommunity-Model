############################################
# assembly_stepwise_ibm.py
############################################
# """
# Step‑wise community assembly.

# -------- assembly_stepwise_ibm.py -------------------------------------
"""
Infinite-pool community assembly for the individual-based model (IBM).

At every “window” we generate
    m = 1 + floor(frac_multi · γ)
brand-new invaders, run the IBM for `window_steps`,
and keep only those invaders that establish.
Extinct residents are pruned permanently (they never return).

Establishment and extinction follow LVMCM-style per-patch detection thresholding:
- Establishment: a newcomer establishes if it reaches >= detection_threshold
  in at least one patch after the window.
- Extinction (pruning): a resident is removed if it has zero patches with
  biomass/abundance >= detection_threshold after the window.

If you set F_sat=None (or np.inf) and max_rounds=None
the loop is in principle unbounded → a true ∞-pool.
"""
from __future__ import annotations
import numpy as np, logging
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD
from assembly_utils import draw_interactions, expand_RC, prune_extinct


# assembly_stepwise_ibm.py  (top-level, after imports)
try:
    import cupy as cp
    def _to_host(a):
        return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
except Exception:
    def _to_host(a):  # CPU-only fallback
        return a
# --------------------------------------------------------------------- #

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

def stepwise_assembly_ibm(
        *,
        base_r        : float = 1.0,
        seed_size     : int   = 50,
        F_sat         : float | None = 1e9,          # ← CHANGED allow None
        max_attempts: int | None = None,   # NEW
        richness_cap: int | None = None,   # NEW
        frac_multi    : float = 0.05,
        window_steps  : int   = 5_000,
        record_step   : int   = 50,
        max_rounds    : int | None ,          # ← CHANGED allow None
        seed          : int   = 0,
        # NEW: allow overriding the detection threshold; default to config.THRESHOLD
        detection_threshold: float | None = None,  # ← NEW (wire from CLI)
        checkpoint_fn = None,                      # ← NEW (call per round)
        # resume support:
        init_r=None, init_C=None, init_N=None,     # ← NEW
        init_attempts: int = 0, init_round: int = -1,  # ← NEW
        **model_kw):

    rng = np.random.default_rng(seed)
    thr = THRESHOLD if detection_threshold is None else detection_threshold  # ← NEW

    # --------- NEW: pick up to 3 fixed patches for αᵢ time series ----------
    _patches_hist = [(0,0), (0, min(Nx-1,1)), (min(Ny-1,1), 0)]   # NEW
    _patches_hist = _patches_hist[:min(3, Ny*Nx)]                 # 
    
    # --------- NEW: round-by-round history buffers -------------------------
    hist_round      = []   # NEW
    hist_attempts   = []   # NEW
    hist_gamma      = []   # NEW
    hist_alpha_bar  = []   # NEW
    hist_alpha_pats = []   # NEW (R, ≤3) richness at the 3 fixed patches
    # NEW: cumulative number of successful invaders (post-window)
    hist_established_cum = []
    est_cum = 0

    # # founder
    # r = np.array([base_r])
    # C = np.array([[1.0]])
    # N = np.zeros((1, Ny, Nx), int)
    # N[0, rng.integers(Ny), rng.integers(Nx)] = 1
    # attempts = 0

    # rounds_iter = range(max_rounds if max_rounds is not None else 10**12)  # ← NEW

    # founder or resume ----------------------------------------------------NEW for resume checkpoint
    if (init_r is not None) and (init_C is not None) and (init_N is not None):  # ← NEW
        r = np.array(init_r, dtype=float, copy=True)
        C = np.array(init_C, dtype=float, copy=True)
        N = np.array(init_N, dtype=int,   copy=True)
        attempts = int(init_attempts)
        start_round = int(init_round) + 1
        log.info(f"[IBM] resume: γ={len(r)} from round {init_round}")
    else:
        r = np.array([base_r], float)
        C = np.array([[1.0]], float)
        N = np.zeros((1, Ny, Nx), int)
        N[0, rng.integers(Ny), rng.integers(Nx)] = 1
        attempts = 0
        start_round = 0

    rounds_iter = range(start_round,
                        (start_round + (max_rounds if max_rounds is not None else 10**12)))  # ∞ if None
    for rnd in rounds_iter:
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
        # NEW: bring GPU array back to CPU before NumPy ops
        N_fin = _to_host(model.N)
        # N_fin = model.N
        attempts += n_cand

        if (max_attempts is not None) and (attempts >= max_attempts):
            log.info(f"[IBM] stop: attempts = {attempts} ≥ invMax")
            break
        if (richness_cap is not None) and (len(r) >= richness_cap):
            log.info(f"[IBM] stop: γ = {len(r)} ≥ cap")
            break

        # keep only established newcomers
        # estab_mask = N_fin[-n_cand:].sum(axis=(1,2)) > 0

        # ── CHANGED: per-patch detection threshold ─────────────────────
        # CHANGED: LVMCM-style establishment — per-patch detection threshold
        # A newcomer establishes if it has at least one patch with N >= thr

        # ---- keep only established newcomers (thresholded) -------------
        estab_mask = (N_fin[-n_cand:] >= thr).any(axis=(1, 2))  # ← CHANGED
        if estab_mask.any():
            keep_new = np.where(estab_mask)[0] + len(r)
            keep_all = np.r_[np.arange(len(r)), keep_new]
            r, C = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
            N    = N_fin[keep_all]
            # log.info(f"[IBM] round {rnd}: {estab_mask.sum()}/{n_cand} "
            #          f"candidates established  → γ={len(r)}")
            log.info(f"[IBM] round {rnd}: {estab_mask.sum()}/{n_cand} "
                       f"candidates established (thr={thr}) → γ={len(r)}")  # ← CHANGED: include thr in log
        else:
            # log.info(f"[IBM] round {rnd}: no candidates established")
            log.info(f"[IBM] round {rnd}: no candidates established (thr={thr})")  # ← CHANGED
            N = N_fin[:len(r)]  # drop all n_cand planes

        # prune extinct residents
        # alive = N.sum(axis=(1,2)) > 0
        # alive = N.sum(axis=(1,2)) >= 3          # need ≥5 individuals in the metacommunity

        # reports established vs pruned and γ before/after:NEW
        gamma_prev = len(r_big)  # before filtering to keep_all/pruning
        established = int(estab_mask.sum())

        # CHANGED: LVMCM-style pruning — species must have at least one patch with N >= thr
        alive = (N >= thr).any(axis=(1, 2))  # ← CHANGED: per-patch detection threshold
        # reports established vs pruned and γ before/after:
        pruned = int((~alive).sum())
        gamma_new = int(alive.sum())
        log.info(f"[IBM] round {rnd}: est={established}/{n_cand}, pruned={pruned}, γ {len(r)}→{gamma_new} (thr={thr})")
        if not alive.all():
            r, C, N, _ = prune_extinct(alive, r, C, N)

        # ---- CHANGED: after pruning, compute richness metrics & record ----
        rich_map = (N >= thr).sum(axis=0)                          # NEW (Ny,Nx)
        alpha_bar_now = float(rich_map.mean())                     # NEW
        alpha_pats_now = [int(rich_map[y, x]) for (y, x) in _patches_hist]  # NEW

        # record history **once per round, after pruning/acceptance**
        est_cum += established
        hist_round.append(int(rnd))                                # NEW
        hist_attempts.append(int(attempts))                        # NEW
        hist_gamma.append(int(len(r)))                             # NEW (post-prune γ)
        hist_alpha_bar.append(alpha_bar_now)                       # NEW
        hist_alpha_pats.append(alpha_pats_now)                     # 
        hist_established_cum.append(int(est_cum)) 

        # ---- checkpoint callback (no copies; saver decides) -------------new
        if checkpoint_fn is not None:  # ← NEW
            occ = (N >= thr).sum(axis=(1, 2))
            checkpoint_fn(dict(
                r=r, C=C, N=N, occ=occ, attempts=attempts,
                gamma=len(r), round=rnd, thr=thr
            ))

        γ = len(r)

        # ── optional saturation stop  ──────────────────────────────────
        if (F_sat not in (None, np.inf)) and attempts >= F_sat * γ:   # ← CHANGED
            log.info(f"[IBM] stop: attempts={attempts} ≥ {F_sat}×γ")
            break

    # ─── bookkeeping ───────────────────────────────────────────────────
    # occ = (N > 0).sum(axis=(1,2))
    # extra = dict(occ_counts=occ)
    # return r, C, N, extra

    # CHANGED: occupancy counts reflect patches above detection threshold
    occ = (N >= thr).sum(axis=(1, 2))  # ← CHANGED
    # extra = dict(occ_counts=occ, detection_threshold=thr)  # ← NEW
    extra = dict(
    occ_counts=occ,
    detection_threshold=thr,
    attempts_total=attempts,      # ← NEW
    round_last=rnd,               # ← NEW
    frac_multi=frac_multi,        # ← NEW
    window_steps=window_steps,    # ← NEW
    seed_size=seed_size,           # ← NEW

    # NEW: true Fig.1a inputs
    ASM_round=np.asarray(hist_round, dtype=np.int64),
    ASM_attempts=np.asarray(hist_attempts, dtype=np.int64),
    ASM_established=np.asarray(hist_established_cum, dtype=np.int64),  # NEW
    ASM_gamma=np.asarray(hist_gamma, dtype=np.int64),
    ASM_alpha_bar=np.asarray(hist_alpha_bar, dtype=np.float32),
    ASM_alpha_patches=np.asarray(hist_alpha_pats, dtype=np.int16),  # shape (R, ≤3)
)
    return r, C, N, extra