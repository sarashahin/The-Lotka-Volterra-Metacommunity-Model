############################################
# run_all_rps.py
############################################
"""
run_all_rps.py  – benchmark RPS OR infinite-pool assembly pipeline
------------------------------------------------------------------
Examples
  3-sp benchmark (IBM)     :  python run_all_rps.py --engine ibm
  3-sp benchmark (PSD2)    :  python run_all_rps.py --engine psd2
  infinite pool (IBM)      :  python run_all_rps.py --engine ibm --pool 120 --tmax 20000
  infinite pool (PSD2)     :  python run_all_rps.py --engine psd2 --pool 120 --tmax 20000
Quick CI smoke-test        :  python run_all_rps.py --dry-run
Outputs → results/{data,plots,movies}/
"""
from __future__ import annotations
import matplotlib, matplotlib.pyplot as plt                      # ①  load Matplotlib *first*
matplotlib.use('Agg')
import argparse, json, time, pathlib, datetime as _dt
import itertools as _it
# import gpu_patch                      # this installs the CuPy shim
# import numpy as np                    # ← already CuPy here      # pyplot sits on MPL core → safe
import scipy   # ok to keep, but avoid scipy.* numerics on GPU path
import gpu_patch
# from  scipy.fft import rfft, rfftreq
from gpu_patch import np, fft
rfft = fft.rfft
rfftfreq = fft.rfftfreq         # cupyx.scipy.fft or real SciPy
# now you may import other stuff
import argparse, json, time, pathlib, datetime as _dt
import numpy as _np

# ─── import GPU patch if USE_GPU=1 in the environment ───────────────────

import sys
import os  # ←—————————————— # NEW



# ─── logging setup ──────────────────────────────────────────────────────
import logging, sys
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


# ─── project modules ─────────────────────────────────────────────────────
import dispersal
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, STEP_SIZE
from models_psd2 import PSD2Model
from models_ibm  import IBMModel
from models_ode  import ODEModel
from assembly_stepwise_psd2 import stepwise_assembly_psd2
from assembly_stepwise_ibm  import stepwise_assembly_ibm
from run_rps_dynamics       import animate_spatial
# visualisation helpers  <‑‑ add these two lines
from utils_vis import make_mosaic
from colour_bank import random_colour_table

# ─────────────────────────────────────────────────────────────────────────

try:
    import cupy
    print("[sanity] backend np:", np.__name__, "| device:", cupy.cuda.runtime.getDevice())
except Exception:
    print("[sanity] backend np:", np.__name__, "(CPU)")
# ─────────────────────────────────────────────────────────────────────────

# run_all_rps.py 
try:
    import cupy as cp
    def _to_host(a):
        return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
except Exception:
    def _to_host(a):
        return a
# --------------------------------------------------------------------- #
# ---------- helpers for training dataset --------------------------------
def summarize_interactions(C):
    """Return in/out degree (excluding diagonal) for C; works with NumPy or CuPy.
       Always returns host (NumPy) arrays (uint16)."""
    try:
        import cupy as cp
        is_cu = isinstance(C, cp.ndarray)
    except Exception:
        cp = None
        is_cu = False

    if is_cu:
        Cnz = (cp.abs(C) > 0)
        cp.fill_diagonal(Cnz, False)
        deg_out = Cnz.sum(axis=0).astype(cp.uint16)
        deg_in  = Cnz.sum(axis=1).astype(cp.uint16)
        return cp.asnumpy(deg_in), cp.asnumpy(deg_out)
    else:
        C = _np.asarray(C)
        Cnz = (_np.abs(C) > 0)
        _np.fill_diagonal(Cnz, False)
        deg_out = Cnz.sum(axis=0).astype(_np.uint16)
        deg_in  = Cnz.sum(axis=1).astype(_np.uint16)
        return deg_in, deg_out
# --------------------------------------------------------------------- #

# ========= NEW: AI helpers ==============================================
def _sanitize_slug(x):
    """Turn floats/None into short filesystem-safe tokens."""
    if x is None: return "NA"
    s = str(x)
    s = s.replace('.', 'p').replace('-', 'm')
    return s

def build_world_tag(base, *, ls, vr, thr, env_seed, grid_y, grid_x, disp, ldd):
    """Unique tag so different worlds don't overwrite each other."""
    return (f"{base}"
            f"_ls{_sanitize_slug(ls)}"
            f"_vr{_sanitize_slug(vr)}"
            f"_thr{_sanitize_slug(thr)}"
            f"_env{_sanitize_slug(env_seed)}"
            f"_grid{grid_y}x{grid_x}"
            f"_dr{_sanitize_slug(disp)}"
            f"_ld{_sanitize_slug(ldd)}")

def topk_interactions(C, k=16):
    """Row-wise top-|C| indices & weights (excluding diagonal)."""
    C = _to_host(C)
    S = C.shape[0]
    A = _np.abs(C).copy()
    _np.fill_diagonal(A, 0.0)
    order = _np.argsort(-A, axis=1)[:, :k]           # (S,k)
    rows  = _np.arange(S)[:, None]
    w     = C[rows, order]                           # signed weights
    return order.astype(_np.int32), w.astype(_np.float32)

def torus_laplacian_edges(Y, X, weight=1.0):
    """4-neighbour periodic grid; returns COO edge list + degrees."""
    edges_u, edges_v = [], []
    for y in range(Y):
        for x in range(X):
            u = y*X + x
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                v = ((y+dy) % Y)*X + ((x+dx) % X)
                edges_u.append(u); edges_v.append(v)
    edges_u = _np.asarray(edges_u, _np.int32)
    edges_v = _np.asarray(edges_v, _np.int32)
    w       = _np.full_like(edges_u, float(weight), dtype=_np.float32)
    degree  = _np.full(Y*X, 4, _np.int32)
    return edges_u, edges_v, w, degree

def rsd_from_presence(P_last):
    """Range size per species = fraction of patches occupied."""
    if P_last is None: return None
    P = _to_host(P_last).reshape(P_last.shape[0], -1)
    return P.mean(axis=1).astype(_np.float32)        # (S,)
# ========================================================================

def take_last_snapshots(B: np.ndarray, t: np.ndarray, k: int = 4):
    """
    Return last k snapshots (time-major), safely if T<k, without NumPy/CuPy
    implicit conversions.
    - B may be NumPy or CuPy
    - t may be NumPy or CuPy (or None)
    """
    if B is None:
        return None, None

    T = int(B.shape[0])
    start = max(0, T - k)

    # Build a HOST (NumPy) index array for any CPU-side uses
    idx_np = _np.arange(start, T, dtype=_np.int64)

    # Decide how to index B and t
    try:
        import cupy as cp
        is_B_cu = isinstance(B, cp.ndarray)
        is_t_cu = (t is not None) and isinstance(t, cp.ndarray)
    except Exception:
        is_B_cu = False
        is_t_cu = False

    if is_B_cu:
        # For B (CuPy), use a CuPy indexer
        import cupy as cp
        idx_cu = cp.asarray(idx_np)
        B_sel = B[idx_cu]
        # For t (could be None/NumPy/CuPy), use a matching indexer
        if t is not None and len(t) == T:
            t_sel = t[idx_cu] if is_t_cu else t[idx_np]
        else:
            t_sel = None
        # Return B on host for downstream saving/plots
        B_sel = cp.asnumpy(B_sel)
        if is_t_cu and t_sel is not None:
            t_sel = cp.asnumpy(t_sel)
    else:
        # B is NumPy → NumPy indexer is fine
        B_sel = B[idx_np]
        if t is not None and len(t) == T:
            t_sel = t[idx_np] if not is_t_cu else _np.asarray(t[idx_np])  # force host
        else:
            t_sel = None

    return B_sel, t_sel


def sample_obs_masks(P_last, budgets=(5, 10, 25), rng=None):
    """For each species, sample a few observation masks at fixed budgets.
       Expects host (NumPy) arrays; returns host arrays."""
    if P_last is None:
        return None
    # Ensure host
    P_last = _to_host(P_last)
    if rng is None:
        rng = _np.random.default_rng(1234)
    S, Y, X = P_last.shape
    out = {}
    flat = P_last.reshape(S, -1)
    for b in budgets:
        M = _np.zeros_like(flat, dtype=_np.uint8)
        for s in range(S):
            pos = _np.where(flat[s] > 0)[0]
            if pos.size:
                take = rng.choice(pos, size=min(b, pos.size), replace=False)
                M[s, take] = 1
        out[f"obs_mask_{b}"] = M.reshape(S, Y, X)
    return out

# ------------------------------------------------------------------


# ─── helpers ─────────────────────────────────────────────────────────────
def dominant_period(t: np.ndarray, x: np.ndarray) -> float:
    if len(t) < 4:
        return float('nan')
    freqs = rfftfreq(len(t), np.mean(np.diff(t)))[1:]
    return float(1.0 / freqs[np.argmax(np.abs(rfft(x))[1:])])

def ensure_dirs(*paths: pathlib.Path) -> None:
    for p in paths:  p.mkdir(parents=True, exist_ok=True)

def _extinction_times_thresholded(B, t, thr_mass):
    """
    First time each species has no patch >= thr_mass.
    B: (T,S,Y,X) biomass; t: (T,); thr_mass scalar.
    Returns (S,) float array of times; -1 if never extinct.
    """
    B = _to_host(B)
    if B is None or B.ndim != 4 or t is None or len(t) != B.shape[0]:
        return None
    # presence per time/species: any patch above thr?
    present = (B >= thr_mass).any(axis=(2, 3))  # (T,S) bool
    extinct_first = (~present).argmax(axis=0)   # position of first False→True transition or 0
    ever_extinct = (~present).any(axis=0)
    out = _np.full(B.shape[1], -1.0, float)
    out[ever_extinct] = _to_host(t)[extinct_first[ever_extinct]]
    return out
# ─────────────────────────────────────────────────────────────────────────

# ─── NEW: robust species-level event times ───────────────────────────────
def species_event_times(P_t: np.ndarray, t: np.ndarray):
    """
    Compute per-species colonization/extinction summary times on the host.
    Returns dict with:
      - T_first_any: first time any patch is occupied (NaN if never)
      - T_last_any : last time any patch is occupied (NaN if never)
      - T_first_ext_after_any: first time with zero occupied after having been occupied (NaN if never)
      - n_recolonizations: number of 0→1→0 cycles (rough)
      - frac_time_occupied: fraction of time steps with any occupancy
    """
    if P_t is None or t is None or len(t) != P_t.shape[0]:
        return None

    P = _to_host(P_t).reshape(P_t.shape[0], P_t.shape[1], -1)  # (T,S,XY)
    any_occ = P.any(axis=2)                                    # (T,S) bool
    T, S = any_occ.shape
    t = _to_host(t).astype(_np.float64)

    first = _np.full(S, _np.nan, _np.float64)
    last  = _np.full(S, _np.nan, _np.float64)
    first_ext_after_any = _np.full(S, _np.nan, _np.float64)
    n_recol = _np.zeros(S, _np.int32)
    frac    = any_occ.mean(axis=0).astype(_np.float32)

    # transitions per species
    for s in range(S):
        series = any_occ[:, s]
        if not series.any():
            continue
        idx = _np.flatnonzero(series)
        first[s] = t[idx[0]]
        last[s]  = t[idx[-1]]

        # find first time it returns to zero after being >0
        post = ~series[idx[0]:]
        if post.any():
            first_ext_after_any[s] = t[idx[0] + post.argmax()]
        # rough recolonizations count (# of 0->1 edges)
        n_recol[s] = int(_np.sum(~series[:-1] & series[1:]))

    return dict(
        T_first_any=first, T_last_any=last,
        T_first_ext_after_any=first_ext_after_any,
        n_recolonizations=n_recol, frac_time_occupied=frac,
    )
# ---------------------------------------------------------------------------


# ---------- NEW: atomic save helper -------------------------------------
def _atomic_save_npz(path: pathlib.Path, **arrays):
    """Atomically save an .npz by writing to a tmp file then renaming.

    Uses a file handle so NumPy does NOT append .npz automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)          # FIX: ensure dir exists
    tmp = path.with_suffix(path.suffix + ".tmp")            # e.g. foo.npz.tmp
    # payload = {k: v for k, v in arrays.items() if v is not None}
    payload = {}

    for k, v in arrays.items():
        if v is None: 
            continue
        payload[k] = _to_host(v)  # ensure CPU

    # Write to the exact tmp path by passing a binary handle (prevents '.npz' re-append)
    with open(tmp, "wb") as fh:                             # FIX: use handle so name is exact
        _np.savez_compressed(fh, **payload)

    os.replace(tmp, path)  # atomic on same filesystem


# ─────────────────────────────────────────────────────────────────────────

def cli(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)   # allow_abbrev=False to avoid issues with long options
    p.add_argument('--engine', type=str, choices=['ibm', 'psd2', 'ode'], default='ibm',
                   help='Simulation engine to use for assembly and dynamics.')
    p.add_argument('--tmax',    type=float, default=1200)
    p.add_argument('--record',  type=float, default=10.)
    p.add_argument('--fps',     type=int,   default=10)
    p.add_argument('--no-movie',action='store_true')
    p.add_argument('--pool',    type=int,
                   help='trigger infinite-pool assembly (number used only for meta)')
    p.add_argument('--assemble-horizon', type=float, default=1_000,
                   help='model-time horizon for each invasion attempt during assembly')
    p.add_argument('--dry-run', action='store_true',
                   help='CI / smoke-test: tmax=100, record=20, skip movies/plots')
    p.add_argument('--skip-ode', action='store_true',
                   help='do not run the ODE reference model')
    p.add_argument('--skip-ibm', action='store_true',
                help='do not run the IBM reference model')    
    p.add_argument('--skip-psd', action='store_true',
            help='do not run the PSD reference model')
    p.add_argument('--skip', nargs='+', choices=['psd', 'ibm', 'ode'],
                   metavar='MODEL', help='space-separated list of models to skip')
    
    # --- IBM specific ---
    p.add_argument('--ibm-frac-multi',  type=float, default=0.05,
                   help='candidate fraction (Jack: 0.05)')
    p.add_argument('--ibm-window-steps', type=int, default=500,
                   help='model steps between invasion attempts')
    p.add_argument('--ibm-F-sat',       type=str, default='None',
                   help="'None' = disable saturation stop")
    p.add_argument('--ibm-max-rounds',  type=str, default='None',
                   help="'None' = unlimited rounds")
    p.add_argument('--ibm-max-attempts', type=str, default='10000',
                   help='hard cap on invasion attempts ( 10000)')
    p.add_argument('--ibm-richness-cap', type=str, default='None',
                   help='gamma cap; usually leave at None')
    p.add_argument('--ibm-record-mode', choices=['full','mean','none'],
               default='mean', help='grid, per-species mean, or nothing')
    
    # --- PSD2 specific ---
    p.add_argument('--psd2-pressure-rate', type=float, default=3e-3,
                   help='Propagule pressure rate for psd2 assembly.')
    
    # -------- NEW: AI export knobs ------------------------------------
    p.add_argument('--C-topk', type=int, default=16,                 # ← NEW
                   help='save top-k interaction partners per species')
    p.add_argument('--coocc-sample', type=int, default=2048,         # ← NEW
                   help='sample this many species-pairs for Jaccard co-occurrence')
    p.add_argument('--world-tag-extra', type=str, default='',        # ← NEW
                   help='optional user tag suffix for outputs')
    
    # ----------------------------------vary dispersal rate / LDD per run from CLI-----------------------------------
    p.add_argument('--disp-rate', type=float, default=None,          # ← NEW
                   help='override dispersal rate (diffusion coefficient)')
    p.add_argument('--ldd-prob', type=float, default=None,           # ← NEW
                   help='override long-distance dispersal probability')

    # ---------------------------------------------------------------------
    
    # ---------- NEW: wire detection threshold + checkpointing/resume -----
    p.add_argument('--ibm-detection-threshold', type=str, default='None',   # ← NEW (used as generic threshold)
                   help="override detection threshold for establish/prune (e.g. 3)")
    p.add_argument('--save-every-rounds', type=int, default=50,             # ← NEW
                   help='checkpoint every K assembly rounds (ibm only)')
    p.add_argument('--save-every-seconds', type=float, default=180.0,       # ← NEW
                   help='checkpoint at least every T seconds (ibm only)')
    p.add_argument('--checkpoint-with-N', action='store_true',              # ← NEW
                   help='include N in checkpoints (ibm only, bigger files)')
    p.add_argument('--resume', type=str, default=None,                      # ← NEW
                   help='path to assembly checkpoint .npz to resume from (ibm only)')
    
    # ---------- NEW: spatially heterogeneous r-field --------------
    
    p.add_argument('--env-length-scale', type=float, default=None,
                help='GRF length scale (None → uniform r)')
    p.add_argument('--env-var-r', type=float, default=None,
                help='Variance of r-field (per species if scalar scales a shared field)')
    p.add_argument('--env-seed-field', type=int, default=None,
                help='Random seed for environment GRF')
    p.add_argument('--save-env-field', action='store_true',
                help='Include ENV_r_field in outputs')
    
    p.add_argument('--fp16-time-series', action='store_true',            # ← NEW
               help='store B_dynamics as float16 in the training .npz')
    p.add_argument('--obs-budgets', type=str, default='5,10,25',         # ← NEW
                help='comma-separated budgets for sparse observation masks')




    return p.parse_args(argv)

# ─────────────────────────────────────────────────────────────────────────
def main(argv=None):
    t0 = time.time()   # ← local timer for this run
    args = cli(argv)
    # --- convert the "None" strings into real None ------------------
    cast = lambda s, typ: None if str(s).lower() == 'none' else typ(s)
    args.ibm_F_sat       = cast(args.ibm_F_sat,       float)
    args.ibm_max_rounds  = cast(args.ibm_max_rounds,  int)
    args.ibm_max_attempts= cast(args.ibm_max_attempts,int)
    args.ibm_richness_cap= cast(args.ibm_richness_cap,int)
    args.ibm_det_thr       = cast(args.ibm_detection_threshold, float)   # ← NEW (used as generic threshold)
    
    # This was inverted, fixed:
    if args.no_movie:
        logging.info("Movies disabled via --no-movie.")
    # args.no_movie = True  # default to no movie, unless --no-movie is set

    # Map the unified  --skip  list onto the individual booleans     # ← NEW
    if getattr(args, 'skip', None):
        args.skip_psd |= 'psd' in args.skip
        args.skip_ibm |= 'ibm' in args.skip
        args.skip_ode |= 'ode' in args.skip

    # ------- dry-run patch ------------------------------------------------
    if args.dry_run:
        args.tmax   = 100
        args.record = 20
        args.no_movie = True
        print("[dry-run] tmax set to 100, record_step to 20, movies disabled.")
    # ----------------------------------------------------------------------

    # overrides for dispersal settings if provided
    if args.disp_rate is not None:                                   # ← NEW
        dispersal.DISPERSAL_RATE = float(args.disp_rate)
    if args.ldd_prob is not None:                                    # ← NEW
        dispersal.LONG_DISTANCE_PROB = float(args.ldd_prob)
    # ----------------------------------------------------------------------------
    # ← NEW parse obs budgets
    try:
        obs_budgets = tuple(int(x) for x in args.obs_budgets.split(',') if x.strip())
    except Exception:
        obs_budgets = (5,10,25)


    # -------------------------------------------------------------------------
    # ---------------------------------------------------------------
    # Shared snapshot schedule (needed by BOTH PSD and IBM sections)  ← NEW
    # ---------------------------------------------------------------
    snap_times = [0,
                  0.25*args.tmax, 0.5*args.tmax, 0.75*args.tmax,
                  0.85*args.tmax, 0.9*args.tmax, 0.95*args.tmax,
                  args.tmax]                                          # ← NEW

    # ---------------------------------------------------------------
    # COMMON RANDOM STARTING MOSAIC  (same biomass everywhere)
    rng      = np.random.default_rng(123)
    B_seed   =  (rng.random((3, NUM_PATCHES_Y, NUM_PATCHES_X)) < 0.5).astype(float) * BODY_MASS
    N_ibm_seed = (B_seed / BODY_MASS).astype(int)         # integer counts for IBM
    # ---------------------------------------------------------------

    # ------- scenario selection ------------------------------------------
    if args.pool is None:                                  # 3-species RPS
        a, b = 1.7, 0.4
        r0 = np.ones(3)
        C0 = np.array([[1,a,b],[b,1,a],[a,b,1]], float)
        tag, use_assembly = 'RPS', False
    else:                                                  # infinite pool
        tag, use_assembly = f'pool{args.pool}', True       # exact size decided later

    # choose colours ONCE, so PSD & IBM panels share them
    # choose colours ONCE, but pick as many as you need
    # max_richness_est = 50 if not use_assembly else 2*args.pool
    # colour_table = random_colour_table(max_richness_est)

    # ← CHANGED: cap to a safe upper bound; colours only needed for optional mosaics
    _rich_guess = 50 if not use_assembly else min(2*int(args.pool), 2000)  # ← NEW cap
    colour_table = random_colour_table(_rich_guess)

                    

    # ------- folders ------------------------------------------------------
    root      = pathlib.Path('results')
    d_data, d_plot, d_mov = root/'data', root/'plots', root/'movies'
    d_ckpt = root/'checkpoints'             # ← NEW
    ensure_dirs(d_data, d_plot, d_mov, d_ckpt)
    # ensure_dirs(d_data, d_plot, d_mov)

    data, meta = {}, {}


    # =====================================================================
    #  Refactored Engine-Specific Logic
    # =====================================================================

    # --- Define generic vars
    r_final = None
    C_final = None
    B_final_assembly = None # This will be N for IBM, B for PSD2
    dynamics_model = None
    t_dynamics = None
    B_dynamics = None
    runtime_s = 0.0
    extra_assembly_outputs = {}
    model_specific_outputs = {} # For psd2's extra return values
    
    # --- Build world tag (must be done before engine block)
    world_tag = build_world_tag(
        base=f"{tag}_{args.engine}" + (f"_{args.world_tag_extra}" if args.world_tag_extra else ""),
        ls=args.env_length_scale, vr=args.env_var_r, thr=args.ibm_det_thr, # Use ibm_det_thr as the generic 'thr'
        env_seed=args.env_seed_field, grid_y=NUM_PATCHES_Y, grid_x=NUM_PATCHES_X,
        disp=float(dispersal.DISPERSAL_RATE), ldd=float(dispersal.LONG_DISTANCE_PROB)
    )
    ckpt_path = d_ckpt / f"{world_tag}_assembly_latest.npz"


    if args.engine == 'ibm':
        if args.skip_ibm:
            logging.info("Skipping IBM engine as requested.")
            sys.exit(0)
            
        # --- IBM Checkpointing & Resume ---
        init_r = init_C = init_N = None
        init_attempts = 0
        init_round = -1
        if args.resume:
            try:
                ck = np.load(args.resume)
                init_r = ck['r']; init_C = ck['C']
                init_N = ck['N'] if 'N' in ck.files else None
                init_attempts = int(ck['attempts']) if 'attempts' in ck.files else 0
                init_round    = int(ck['round']) if 'round' in ck.files else -1
                logging.info(f"[resume] loaded {args.resume} with γ={len(init_r)}, round={init_round}, attempts={init_attempts}")
            except Exception as e:
                logging.error(f"Failed to load resume checkpoint {args.resume}: {e}")
                sys.exit(1)
        
        # --- IBM Checkpoint Callback ---
        last_save_t = time.time()
        def on_round(state):
            nonlocal last_save_t
            do_round = (state['round'] % max(1, args.save_every_rounds) == 0)
            do_time  = (time.time() - last_save_t) >= max(1.0, args.save_every_seconds)
            if (do_round or do_time) and state['round'] > init_round: # Dont save on first round if resuming
                payload = dict(
                    r=state['r'], C=state['C'],
                    occ_counts=state['occ'],
                    attempts=state['attempts'], round=state['round'],
                    gamma=_np.int64(state['gamma']),
                    detection_threshold=_np.float64(state['thr'])
                )
                if args.checkpoint_with_N:
                    payload['N'] = state['N']
                _atomic_save_npz(ckpt_path, **payload)
                last_save_t = time.time()
                logging.info(f"[checkpoint] γ={state['gamma']} round={state['round']} attempts={state['attempts']} → {ckpt_path.name}")
        
        if use_assembly:
            logging.info("Running assembly with IBM engine...")
            r_final, C_final, N_final_assembly, extra_assembly_outputs = stepwise_assembly_ibm(
                window_steps=args.ibm_window_steps,
                record_step=int(args.record/STEP_SIZE),
                seed_size=5,
                F_sat=args.ibm_F_sat,
                max_rounds=args.ibm_max_rounds,
                max_attempts=args.ibm_max_attempts,            
                richness_cap=args.ibm_richness_cap,       
                frac_multi=args.ibm_frac_multi, 
                seed=1,
                detection_threshold=args.ibm_det_thr,
                checkpoint_fn=on_round,
                init_r=init_r, init_C=init_C, init_N=init_N,
                init_attempts=init_attempts, init_round=init_round,
                length_scale=args.env_length_scale,
                var_r=args.env_var_r,
                seed_field=args.env_seed_field
            )
            B_final_assembly = N_final_assembly # Store N here
        else:
            logging.info("Running 3-species RPS with IBM engine...")
            r_final, C_final = r0, C0
            B_final_assembly = N_ibm_seed # This is the (3,Y,X) seed
            extra_assembly_outputs = {}

        # --- IBM Dynamics Run ---
        logging.info("Running dynamics with IBM engine...")
        dynamics_model = IBMModel(r_final, C_final, initial_N=B_final_assembly,
                            nsteps=int(args.tmax/STEP_SIZE),
                            record_step=int(args.record/STEP_SIZE),
                            record_mode=args.ibm_record_mode,
                            dispersal_type='propagule', seed=1,
                            length_scale=args.env_length_scale,
                            var_r=args.env_var_r,
                            seed_field=args.env_seed_field)
        
        t_run0 = time.time()
        B_dynamics = dynamics_model.run() # This is (T,S,Y,X) or (T,S)
        runtime_s = time.time() - t_run0
        
        B_dynamics = _to_host(B_dynamics)
        T_rec = getattr(dynamics_model, "nrecords", None)
        if T_rec is None:
            T_rec = B_dynamics.shape[0] if B_dynamics is not None else int(args.tmax // args.record)
        t_dynamics = _np.arange(1, T_rec + 1, dtype=_np.int64) * _np.int64(args.record)
        
        # Store N_final_assembly in the dict to save it
        model_specific_outputs['N_final_assembly'] = B_final_assembly


    elif args.engine == 'psd2':
        if args.skip_psd:
            logging.info("Skipping PSD2 engine as requested.")
            sys.exit(0)
        
        if args.resume:
            logging.warning(f"Resume is not supported for 'psd2' engine. Starting fresh.")
        
        if use_assembly:
            logging.info("Running assembly with PSD2 engine...")
            # Note: assembly_stepwise_psd2 does not support heterogeneous r_field
            if args.env_length_scale is not None:
                logging.warning("PSD2 assembly does not support heterogeneous environments. Using base_r=1.0.")
            
            r_final, C_final, extra_assembly_outputs = stepwise_assembly_psd2(
                base_r=1.0, 
                pressure_rate=args.psd2_pressure_rate,
                window_time=args.assemble_horizon,
                record_step=args.record,
                F_sat=args.ibm_F_sat, # Reuse
                frac_multi=args.ibm_frac_multi, # Reuse
                max_rounds=args.ibm_max_rounds, # Reuse
                seed=1
            )
            B_final_assembly = extra_assembly_outputs.get('B_seed') # This is (S,Y,X) biomass
        else:
            logging.info("Running 3-species RPS with PSD2 engine...")
            r_final, C_final = r0, C0
            B_final_assembly = B_seed # This is the (3,Y,X) seed
            extra_assembly_outputs = {}
        
        # --- PSD2 Dynamics Run ---
        logging.info("Running dynamics with PSD2 engine...")
        dynamics_model = PSD2Model(r_final, C_final, initial_B=B_final_assembly,
                                tmax=args.tmax, record_step=args.record,
                                dispersal_type='propagule', seed=1,
                                length_scale=args.env_length_scale,
                                var_r=args.env_var_r,
                                seed_field=args.env_seed_field)
        
        t_run0 = time.time()
        t_psd2, B_psd2, w_psd2, pc_psd2, g_psd2, inv_psd2, est_psd2 = dynamics_model.run()
        runtime_s = time.time() - t_run0
        
        t_dynamics = _to_host(t_psd2)
        B_dynamics = _to_host(B_psd2) # (T,S,Y,X)
        
        model_specific_outputs = {
            'PSD2_wait': _to_host(w_psd2),
            'PSD2_pclock': _to_host(pc_psd2),
            'PSD2_growth': _to_host(g_psd2),
            'PSD2_invasion': _to_host(inv_psd2),
            'PSD2_est_prob': _to_host(est_psd2),
            'B_final_assembly': _to_host(B_final_assembly)
        }

    elif args.engine == 'ode':
        if args.skip_ode:
            logging.info("Skipping ODE engine as requested.")
            sys.exit(0)
            
        if use_assembly:
            logging.error("Assembly with 'ode' engine is not supported. Halting.")
            sys.exit(1)
        else:
            logging.info("Running 3-species RPS with ODE engine...")
            r_final, C_final = r0, C0
            # B_final_assembly is None, ODEModel initializes itself from config
            B_final_assembly = None 
            extra_assembly_outputs = {}

        # --- ODE Dynamics Run ---
        logging.info("Running dynamics with ODE engine...")
        dynamics_model = ODEModel(r_final, C_final,
                                tmax=args.tmax, record_step=args.record,
                                dispersal_type='propagule', seed=1,
                                length_scale=args.env_length_scale,
                                var_r=args.env_var_r,
                                seed_field=args.env_seed_field)

        t_run0 = time.time()
        t_ode, B_ode = dynamics_model.run()
        runtime_s = time.time() - t_run0
        
        t_dynamics = _to_host(t_ode)
        B_dynamics = _to_host(B_ode)
        
        # ODE model has no extra outputs
        model_specific_outputs = {}

    else:
        logging.error(f"Unknown engine: {args.engine}")
        sys.exit(1)


    # =====================================================================
    #  GENERIC POST-PROCESSING & SAVING
    # =====================================================================
    logging.info("Post-processing and saving results...")

    # Ensure assembly outputs are on host
    r_final = _to_host(r_final)
    C_final = _to_host(C_final)
    if B_final_assembly is not None:
         B_final_assembly = _to_host(B_final_assembly)

    # NEW: interaction magnitude summaries (on host)
    C_host = _to_host(C_final)
    Aabs = _np.abs(C_host)
    _np.fill_diagonal(Aabs, 0.0)
    alpha_mu  = Aabs.mean(axis=1).astype(_np.float32)
    alpha_med = _np.median(Aabs, axis=1).astype(_np.float32)
    alpha_p95 = _np.quantile(Aabs, 0.95, axis=1).astype(_np.float32)


    # capture environment from model (host array)
    ENV_r_field = _to_host(getattr(dynamics_model, "r_field", None))

    # --- species features from r-field --------------------------------add species r-stats
    r_mean = r_std = None
    if ENV_r_field is not None and ENV_r_field.ndim == 3:
        r_mean = ENV_r_field.mean(axis=(1,2)).astype(_np.float32)  # (S,)
        r_std  =  ENV_r_field.std (axis=(1,2)).astype(_np.float32)  # (S,)

    # ---------------------------------------------------------------------
    # --- NEW: compute extinction times BEFORE adding to train_out ------------
    extinction_step_times = None
    if (B_dynamics is not None) and (B_dynamics.ndim == 4):
        thr_for_ext = args.ibm_det_thr if (args.ibm_det_thr is not None) \
            else float(extra_assembly_outputs.get('detection_threshold', 1.0))
        extinction_step_times = _extinction_times_thresholded(
            B_dynamics, t_dynamics, thr_mass=thr_for_ext * BODY_MASS
    )



    # ---------- TRAINING DATA EXPORT (NEW) -----------------------------------
    B_last = None; P_last_final = None; B_k = None; t_k = None
    
    # Use generic threshold from --ibm-detection-threshold arg
    thr = args.ibm_det_thr if (args.ibm_det_thr is not None) \
        else float(extra_assembly_outputs.get('detection_threshold', 1.0))

    if (B_dynamics is not None) and (B_dynamics.ndim == 4):
        B_last_raw = B_dynamics[-1] # (S,Y,X)
        B_k, t_k = take_last_snapshots(B_dynamics, t_dynamics, k=4)
        if B_k is not None:
            B_k = B_k.astype(_np.float16)
        
        P_last_final = (B_last_raw >= (thr * BODY_MASS)).astype(_np.uint8)
        B_last = B_last_raw.astype(_np.float16)
        
    elif B_final_assembly is not None:
        if args.engine == 'ibm': # B_final_assembly is N
            B_last = (B_final_assembly * BODY_MASS).astype(_np.float16)
            P_last_final = (B_final_assembly >= int(thr)).astype(_np.uint8)
        else: # For psd2, B_final_assembly is B
            B_last = B_final_assembly.astype(_np.float16)
            P_last_final = (B_last >= (thr * BODY_MASS)).astype(_np.uint8)
            

    P_init_assembly = None
    if B_final_assembly is not None:
        if args.engine == 'ibm': # B_final_assembly is N
            P_init_assembly = (B_final_assembly >= int(thr)).astype(_np.uint8)
        else: # psd2, B_final_assembly is B
            P_init_assembly = (B_final_assembly >= (thr * BODY_MASS)).astype(_np.uint8)

    P_t = None
    if (B_dynamics is not None) and (B_dynamics.ndim == 4):
        P_t = (B_dynamics >= (thr * BODY_MASS)).astype(_np.uint8)

    # ---------- temporal summaries (guarded) ----------
    RSD_t = gamma_t = T_first_occ = persist_steps = None
    if P_t is not None:
        # (T,S) fraction of patches occupied
        RSD_t = P_t.reshape(P_t.shape[0], P_t.shape[1], -1).mean(axis=2).astype(_np.float32)

        # (T,) regional richness (any patch occupied)
        gamma_t = P_t.reshape(P_t.shape[0], P_t.shape[1], -1).any(axis=2).sum(axis=1).astype(_np.int32)

        # per-patch colonization time & persistence length
        T_idx = t_dynamics  # (T,)
        S, Y, X = P_t.shape[1:]
        T_first_occ  = _np.full((S, Y, X), -1, dtype=_np.int32)
        persist_steps= _np.zeros((S, Y, X), dtype=_np.int32)
        pt = P_t.transpose(1,2,3,0)  # (S,Y,X,T)
        for s in range(S):
            for y in range(Y):
                for x in range(X):
                    series = pt[s,y,x]
                    if series.any():
                        first = series.argmax()
                        T_first_occ[s,y,x]  = int(T_idx[first])
                        persist_steps[s,y,x]= int(series.sum())

    # -----------------------------------------------------------------------------------
    # NEW: species-level turnover features
    SP_events = species_event_times(P_t, t_dynamics) if P_t is not None else None

    # ---------- obs masks & range size from final labels ----------
    obs_masks = sample_obs_masks(P_last_final, budgets=obs_budgets) if P_last_final is not None else None

    RSD = rsd_from_presence(P_last_final) if P_last_final is not None else None

    # ---------- co-occurrence (sampled) ----------
    # ---------- co-occurrence (sampled, stratified by prevalence) ----------
    # *** NEW: any-time presence & prevalence ***
    P_any = None; prev_final = None; prev_any = None; w_invprev_final = None; w_invprev_any = None
    if P_t is not None:
        # presence at any time across the run
        P_any = P_t.any(axis=0).astype(_np.uint8)  # (S,Y,X)
        prev_any = P_any.reshape(P_any.shape[0], -1).mean(axis=1).astype(_np.float32)  # (S,)
    if P_last_final is not None:
        prev_final = P_last_final.reshape(P_last_final.shape[0], -1).mean(axis=1).astype(_np.float32)

    # suggested inverse-prevalence weights (rare species ↑)
    _eps = _np.float32(1e-6)
    if prev_final is not None:
        w_invprev_final = (1.0 / (prev_final + _eps)).astype(_np.float32)
        w_invprev_final /= w_invprev_final.mean()  # normalize ~1
    if prev_any is not None:
        w_invprev_any = (1.0 / (prev_any + _eps)).astype(_np.float32)
        w_invprev_any /= w_invprev_any.mean()

    # *** NEW: co-occurrence on any-time presence (stratified) ***
    CO_J_any = None
    if P_any is not None and int(args.coocc_sample) > 0:
        rng = _np.random.default_rng(123)
        S = P_any.shape[0]
        pr_any = P_any.reshape(S, -1).astype(bool)
        prev_a = pr_any.mean(axis=1)
        bins = _np.clip((prev_a * 10).astype(int), 0, 9)
        pairs_per_bin = max(1, args.coocc_sample // 12)
        js_any = []
        for a_bin in range(10):
            for b_bin in range(a_bin, 10):
                cand_a = _np.flatnonzero(bins == a_bin)
                cand_b = _np.flatnonzero(bins == b_bin)
                if cand_a.size == 0 or cand_b.size == 0:
                    continue
                # draw equal-length aligned samples
                L = min(pairs_per_bin, max(1, cand_a.size), max(1, cand_b.size))
                A = rng.choice(cand_a, size=L, replace=True)
                B = rng.choice(cand_b, size=L, replace=True)

                # only drop self-pairs when sampling within the same bin
                if a_bin == b_bin:
                    keep = (A != B)               # 1-D boolean mask, same length as A and B
                    A, B = A[keep], B[keep]
                if A.size == 0:                    # nothing left to score in this (a_bin,b_bin)
                    continue

                inter = (pr_any[A] & pr_any[B]).sum(axis=1)
                uni   = (pr_any[A] | pr_any[B]).sum(axis=1)
                js_any.append((inter / _np.maximum(1, uni)).astype(_np.float32))

        if js_any:
            CO_J_any = _np.concatenate(js_any, axis=0)

    # ---------------------------------------------------------------
    # ---------- interactions & grid ----------
    deg_in, deg_out = summarize_interactions(C_final)
    C_top_idx, C_top_w = topk_interactions(C_final, k=max(1, int(args.C_topk)))
    G_u, G_v, G_w, G_deg = torus_laplacian_edges(NUM_PATCHES_Y, NUM_PATCHES_X)

    # -----------------------------------------------------------------------------
    # NEW: QA checks
    if B_dynamics is not None:
        assert B_dynamics.shape[0] == len(t_dynamics), f"B_dynamics/t_dynamics mismatch: {B_dynamics.shape[0]} vs {len(t_dynamics)}"
    if B_k is not None and t_k is not None:
        assert len(t_k) == B_k.shape[0], "B_lastK/t_lastK mismatch"
    if P_t is not None:
        assert P_t.shape[0] == len(t_dynamics), "P_t/t_dynamics mismatch"
    # ---------------------------------------------------------------

    dynamics_B_to_save = None
    if B_dynamics is not None:
        # Handle different record modes
        if args.engine == 'ibm' and args.ibm_record_mode != 'full':
             dynamics_B_to_save = B_dynamics.astype(_np.float16) # Already (T,S) or None
        elif B_dynamics.ndim == 4: # Full dynamics run
             dynamics_B_to_save = B_dynamics.astype(_np.float16) if args.fp16_time_series else B_dynamics
        else:
             dynamics_B_to_save = B_dynamics # Fallback


    # ---------- build payload ONCE, then save ----------
    train_out = {
        "B_last": B_last,
        "P_last_final": P_last_final,
        "P_init_assembly": P_init_assembly,
        "P_t": P_t,
        "B_lastK": B_k,
        "t_lastK": t_k,
        "deg_in": deg_in, "deg_out": deg_out,
        "gamma": _np.int64(len(r_final)),
        "Y": _np.int64(NUM_PATCHES_Y),
        "X": _np.int64(NUM_PATCHES_X),

        "BODY_MASS": _np.float32(BODY_MASS),

        # *** FIX: read from 'dispersal' module so CLI overrides are recorded ***
        "DISPERSAL_RATE": _np.float32(float(dispersal.DISPERSAL_RATE)),
        "LONG_DISTANCE_PROB": _np.float32(float(dispersal.LONG_DISTANCE_PROB)),

        "detection_threshold": _np.float32(thr),
        "window_steps": _np.int64(extra_assembly_outputs.get("window_steps", args.ibm_window_steps if args.engine == 'ibm' else -1)),
        "seed_size": _np.int64(extra_assembly_outputs.get("seed_size", -1)),
        "attempts_total": _np.int64(extra_assembly_outputs.get("attempts_total", -1)),
        "round_last": _np.int64(extra_assembly_outputs.get("round_last", -1)),
        "seed": _np.int64(1),

        "t_dynamics": t_dynamics,
        "B_dynamics": dynamics_B_to_save,
        "extinction_step_times": extinction_step_times,
        "runtime_s": _np.float32(runtime_s),

        # extras for AI
        "C_topk_idx": C_top_idx, "C_topk_w": C_top_w,
        "GRID_u": G_u, "GRID_v": G_v, "GRID_w": G_w, "GRID_deg": G_deg,
        "r_base": r_final.astype(_np.float32),
        "r_mean": r_mean, "r_std": r_std,
        "RSD": RSD,
        "RSD_t": RSD_t, "gamma_t": gamma_t,
        "T_first_occ": T_first_occ, "persist_steps": persist_steps,

        # *** NEW: species event summaries (now saved) ***
        "SP_T_first_any": SP_events["T_first_any"] if SP_events else None,
        "SP_T_last_any":  SP_events["T_last_any"] if SP_events else None,
        "SP_T_first_ext_after_any": SP_events["T_first_ext_after_any"] if SP_events else None,
        "SP_n_recolonizations": SP_events["n_recolonizations"] if SP_events else None,
        "SP_frac_time_occupied": SP_events["frac_time_occupied"] if SP_events else None,

        # *** NEW: any-time presence & prevalence/weights ***
        "P_any": P_any,
        "prevalence_final": prev_final,
        "prevalence_any": prev_any,
        "w_invprev_final": w_invprev_final,
        "w_invprev_any": w_invprev_any,
    }
    # ----------------New---------------------------
    # attach alpha summaries here  # *** FIX ***
    train_out["alpha_abs_mean"]   = alpha_mu
    train_out["alpha_abs_median"] = alpha_med
    train_out["alpha_abs_p95"]    = alpha_p95

    # *** NEW: any-time co-occurrence ***
    if CO_J_any is not None:
        train_out["COOCC_Jaccard_anytime"] = CO_J_any
    
    # Add model-specific outputs (e.g., PSD2 wait times, IBM final N)
    train_out.update(model_specific_outputs)


    # NEW: optionally include environment in training file (unchanged idea)
    if args.save_env_field and ENV_r_field is not None:
        train_out.update({
            "ENV_r_field": ENV_r_field.astype(_np.float32),
            "ENV_length_scale": _np.float32(args.env_length_scale if args.env_length_scale is not None else -1.0),
            "ENV_var_r": _np.float32(args.env_var_r if args.env_var_r is not None else 0.0),
            "ENV_seed_field": _np.int64(args.env_seed_field if args.env_seed_field is not None else -1),
            "ENV_type": _np.array(["sqrt-exp GRF"], dtype=object),
        })
    
    # attach assembly history (single loop)
    for key in ("ASM_round", "ASM_attempts", "ASM_established",
                "ASM_gamma", "ASM_alpha_bar", "ASM_alpha_patches"):
        
        if key in extra_assembly_outputs:
            train_out[key] = _to_host(extra_assembly_outputs[key])

    if obs_masks:
        train_out.update(obs_masks)

    train_path = d_data / f"{world_tag.lower()}_training.npz"
    _atomic_save_npz(train_path, **train_out)
    logging.info(f"[save][train] → {train_path}")

    # ----------------------------------------------------------------------


    if not args.no_movie:
        # Check for full dynamics run before animating
        if B_dynamics is not None and B_dynamics.ndim == 4:
            animate_spatial(B_dynamics, f'{args.engine.upper()} {tag}', str(d_mov/f'{args.engine.upper()}_{tag}.mp4'), args.fps)
        elif args.engine == 'ibm' and args.ibm_record_mode != 'full':
            logging.warning(f"Cannot generate movie for IBM with record_mode='{args.ibm_record_mode}'. Use 'full'.")
        else:
            logging.warning("Cannot generate movie. Full dynamics trajectory (T,S,Y,X) not available.")


    # =====================================================================
    #  SAVE final dataset (meta)
    # =====================================================================
    # right before saving fout:
    total_runtime_s = time.time() - t0
    meta.update(dict(
        scenario   = tag,
        engine     = args.engine, # <-- NEW: record engine
        timestamp  = _dt.datetime.now(_dt.UTC).isoformat(timespec='seconds').replace('+00:00','Z'),
        grid       = f"{NUM_PATCHES_Y}×{NUM_PATCHES_X}",
        dispersal = dict(
        DISPERSAL_RATE=float(dispersal.DISPERSAL_RATE),
        LONG_DISTANCE_PROB=float(dispersal.LONG_DISTANCE_PROB),
    ),
        body_mass  = BODY_MASS,
        runtime_s  = float(total_runtime_s),   # NEW
    ))
    # Save meta to a separate JSON, and the main data to NPZ (which is already done)
    # fout = d_data / f'{world_tag.lower()}_dataset.npz'
    # np.savez_compressed(fout, meta=json.dumps(meta, indent=2), **data)
    
    # Save meta as JSON
    meta_path = d_data / f"{world_tag.lower()}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print('[save] →', train_path)
    print('[save] →', meta_path)
    print(f'[done] finished successfully in {total_runtime_s:.1f}s.')


    # =====================================================================
    #  QUICK-LOOK PLOT
    # =====================================================================
    if not args.no_movie and not args.dry_run:
        plt.figure(figsize=(8,6))
        
        # Plot mean biomass from the dynamics run
        if B_dynamics is not None:
            mean_biomass = None
            if B_dynamics.ndim == 4: # (T,S,Y,X)
                mean_biomass = B_dynamics.mean(axis=(1,2,3))
            elif B_dynamics.ndim == 2: # (T,S) - e.g., from ibm-record-mode=mean
                mean_biomass = B_dynamics.mean(axis=1)
            
            if mean_biomass is not None:
                 plt.plot(t_dynamics, mean_biomass, label=f'{args.engine.upper()} Mean Biomass')

        # The old logic for periods doesn't fit well with generic vars
        # txt = '   '.join(f"{k}:{v['period']:.1f}"
        #                  for k,v in meta.items()
        #                  if isinstance(v, dict) and 'period' in v)
        plt.title(f'{tag} – {args.engine.upper()} Engine')
        plt.xlabel('time'); plt.ylabel('mean biomass'); plt.legend()
        plt.tight_layout()
        plt.savefig(d_plot/f'{tag}_{args.engine}_means.png', dpi=150); plt.close()

    print(f'[done] finished successfully. Engine: {args.engine}')

# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
# if __name__ == "__main__" and "--quick-exit" in sys.argv:
#     print("Quick-exit sanity-check OK"); sys.exit(0)
    t0 = time.time()
    main()
    print(f'⏱  {time.time()-t0:.1f}s')
