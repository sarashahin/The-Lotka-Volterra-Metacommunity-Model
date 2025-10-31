############################################
# simulation_utils.py
############################################
"""
Utility functions for simulation post-processing, data analysis, and file I/O.
"""
import os
import time
import numpy as _np
from pathlib import Path

# Import constants needed for translation
from config import BODY_MASS

# Attempt to import CuPy for GPU operations
try:
    import cupy as cp
    _GPU_ENABLED = True
except ImportError:
    _GPU_ENABLED = False

# --- Host/Device Data Transfer ---

def to_host(arr):
    """Ensure an array is a NumPy array on the host CPU."""
    if _GPU_ENABLED and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

# --- State Translation Functions ---

def translate_psd_state_to_ibm(psd_state: tuple, rng: _np.random.Generator = None) -> _np.ndarray:
    """
    Translates a PSD2 state (biomass) to an IBM state (individual counts).

    For each species in each patch, the number of individuals is sampled from a
    Poisson distribution whose mean is given by (Biomass / BODY_MASS).

    Args:
        psd_state: A tuple (B, W, PC) representing the PSD2 state.
        rng: A NumPy random number generator for reproducibility. If None, a new
             one is created.

    Returns:
        A NumPy array of integer counts `N` representing the corresponding IBM state.
    """
    if rng is None:
        rng = _np.random.default_rng()

    biomass, _, _ = psd_state
    biomass_host = to_host(biomass)
    mean_individuals = biomass_host / BODY_MASS #
    mean_individuals[mean_individuals < 0] = 0
    N = rng.poisson(mean_individuals).astype(int)
    return N

def translate_ibm_state_to_psd(ibm_state_N: _np.ndarray) -> tuple:
    """
    Performs a PROVISIONAL translation of an IBM state to a PSD2 state.

    - logB is set based on N * BODY_MASS.
    - Patches with N=0 are provisionally set to the 'S' (waiting) state.
    - This state must be refined later using `local_growth` information.

    Args:
        ibm_state_N: A NumPy array of integer counts `N`.

    Returns:
        A provisional PSD2 state tuple (B, W, PC).
    """
    N = to_host(ibm_state_N)
    S, Ny, Nx = N.shape

    # Calculate logB based on N
    # Use a very small biomass for extinct patches
    B = _np.where(N > 0,
                  N * BODY_MASS, #
                  BODY_MASS / 10000) #
    
    # Provisionally set waiting flags and clocks
    # N > 0 -> Active 'D' state (waiting=False, pclock=1.0)
    # N == 0 -> Waiting 'S' state (waiting=True, pclock=random)
    W = (N == 0)
    PC = _np.where(W,
                   _np.log(_np.random.rand(S, Ny, Nx)), # New random clock
                   1.0)                                # Active state clock
                   
    return (B, W, PC)

# --- Data Analysis & Feature Extraction ---
# ... (the rest of the file remains unchanged) ...
def dominant_period(t, x):
    t, x = to_host(t), to_host(x)
    if len(t) < 4: return float('nan')
    from scipy.fft import rfft, rfftreq
    dt = _np.mean(_np.diff(t)); freqs = rfftreq(len(t), dt)[1:]
    if not len(freqs): return float('nan')
    spec = _np.abs(rfft(x))[1:]; return 1.0 / freqs[_np.argmax(spec)]

def summarize_interactions(C):
    C_host = to_host(C); Cnz = (_np.abs(C_host) > 0); _np.fill_diagonal(Cnz, False)
    return Cnz.sum(axis=1).astype(_np.uint16), Cnz.sum(axis=0).astype(_np.uint16)

def topk_interactions(C, k=16):
    C_host = to_host(C); S = C_host.shape[0]; A = _np.abs(C_host).copy()
    _np.fill_diagonal(A, 0.0); order = _np.argsort(-A, axis=1)[:, :k]
    rows = _np.arange(S)[:, None]; weights = C_host[rows, order]
    return order.astype(_np.int32), weights.astype(_np.float32)

def species_event_times(presence_t, t):
    P_t, t_h = to_host(presence_t), to_host(t)
    if P_t is None or t_h is None or len(t_h) != P_t.shape[0]: return None
    any_occ = P_t.reshape(P_t.shape[0], P_t.shape[1], -1).any(axis=2)
    T, S = any_occ.shape
    times = {"T_first_any": _np.full(S, _np.nan, _np.float64), "T_last_any": _np.full(S, _np.nan, _np.float64), "T_first_ext_after_any": _np.full(S, _np.nan, _np.float64), "n_recolonizations": _np.zeros(S, _np.int32), "frac_time_occupied": any_occ.mean(axis=0).astype(_np.float32)}
    for s in range(S):
        series = any_occ[:, s]
        if not series.any(): continue
        idx = _np.flatnonzero(series)
        times["T_first_any"][s] = t_h[idx[0]]; times["T_last_any"][s] = t_h[idx[-1]]
        post_first_occ = ~series[idx[0]:]; 
        if post_first_occ.any(): times["T_first_ext_after_any"][s] = t_h[idx[0] + post_first_occ.argmax()]
        times["n_recolonizations"][s] = int(_np.sum(~series[:-1] & series[1:]))
    return times

def ensure_dirs(*paths: Path):
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def _sanitize_slug(x):
    if x is None: return "NA"
    return str(x).replace('.', 'p').replace('-', 'm')

def build_world_tag(base, ls, vr, thr, env_seed, grid_y, grid_x, disp, ldd):
    return (f"{base}_ls{_sanitize_slug(ls)}_vr{_sanitize_slug(vr)}_thr{_sanitize_slug(thr)}_env{_sanitize_slug(env_seed)}_grid{grid_y}x{grid_x}_dr{_sanitize_slug(disp)}_ld{_sanitize_slug(ldd)}")

def atomic_save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp_{int(time.time()*1e6)}")
    payload = {k: to_host(v) for k, v in arrays.items() if v is not None}
    with open(tmp_path, "wb") as fh: _np.savez_compressed(fh, **payload)
    os.replace(tmp_path, path)
