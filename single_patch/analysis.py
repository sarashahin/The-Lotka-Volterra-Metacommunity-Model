
############################################
# analysis.py
############################################

"""
Analysis routines: 
 - plotting (matplotlib),
 - histograms,
 - alpha diversity, 
 - turnover rates,
 - covariance,
 - data saving for advanced/Plotly usage later
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils import mav, bav, mean_se
import config_utils
from config import (
    BODY_MASS,
    MORTALITY_RATE,
    STEP_SIZE,
    RECORDING_STEP_SIZE,
    TMAX, INV, RTOL, ATOL)

logger = logging.getLogger(__name__)

def alpha_diversity(trajectory, threshold=None):
    """
    Count how many species have biomass > threshold at each time record.
    """
    if threshold==None:
        threshold = THRESHOLD
    return np.sum(trajectory > threshold, axis=1)

def turnover_rate(trajectory, recording_step):
    """
    Calculate changes in presence/absence above threshold 
    from one record to the next.
    """
    presence = (trajectory > THRESHOLD).astype(int)
    diffs = np.abs(np.diff(presence, axis=0))
    changes = np.sum(diffs, axis=1)
    rate = changes / float(recording_step)
    return rate    
##############################################################################

def plot_total_biomasses(trajectories, labels, title, filename):
    """
    Multi-trajectory line plot: total biomass versus time.
    """
    plt.figure(figsize=(10,6))
    for traj, label in zip(trajectories, labels):
        total_biomass = np.sum(traj, axis=1)
        plt.plot(total_biomass, label=label)
    plt.xlabel("Time Index")
    plt.ylabel("Total Biomass")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def plot_trajectories(trajectories, labels, title, filename_base):
    """
    Multi-trajectory line plots: biomasses versus time for each model.
    """
    for traj, label in zip(trajectories, labels):
        plt.figure(figsize=(10,6))
        for i in range(traj.shape[1]):
            plt.plot(traj[:, i])
        plt.xlabel("Time Index")
        plt.ylabel("Biomass")
        plt.title(label)
        filename = label + "_" + filename_base
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved figure {filename}")
        

# ───────────────────────────── histogram helpers ─────────────────────────
def _auto_bins(data: np.ndarray, rule: str = "FD", max_bins: int = 200) -> int:
    """Scott / Freedman–Diaconis bin selector (upper-capped)."""
    if data.size < 5:
        return 5
    if rule.upper().startswith("S"):
        # Scott
        bw = np.histogram_bin_edges(data, bins="scott")
    else:
        # Freedman-Diaconis
        bw = np.histogram_bin_edges(data, bins="fd")
    # np.histogram_bin_edges returns the EDGE array → #bins = len(edges)-1
    n_bins = len(bw) - 1
    return max(5, min(n_bins, max_bins))

def _draw_hist(ax, biomass: np.ndarray, label: str,
               bins="auto", color=None, **kw):
    biomass = biomass[biomass >= BODY_MASS]
    if biomass.size == 0:
        return None
    x = np.log10(biomass)
    if bins == "auto" or (isinstance(bins, tuple) and bins[0] == "auto"):
        rule = "FD" if bins == "auto" else bins[1]
        bins = _auto_bins(x, rule=rule)
    return ax.hist(x, bins=bins, density=True, alpha=.45,
                   label=label, color=color, **kw)[0]


def histogram_biomass(trajectories, labels, filename,
                      *, bins="auto", comparison: bool = False, burn_in_frac: float = 0.0):
    """
    Parameters
    ----------
    trajectories : list[np.ndarray]   flattened biomass arrays
    labels       : list[str]
    bins         : int | "auto" | ("auto","Scott") | ("auto","FD")
    comparison   : if True, the first two curves are compared via KS statistic
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    curves = []
    for traj, lab in zip(trajectories, labels):
        # --- discard the first burn_in_frac of records -----------------
        if burn_in_frac > 0:
            cut = int(traj.shape[0] * burn_in_frac)
            traj = traj[cut:]
        n = _draw_hist(ax, traj.flatten(), lab, bins=bins)
        curves.append(n)

    if comparison and len(curves) >= 2 and curves[0] is not None and curves[1] is not None:
        cdf1 = np.cumsum(curves[0]) / curves[0].sum()
        cdf2 = np.cumsum(curves[1]) / curves[1].sum()
        ks   = np.max(np.abs(cdf1 - cdf2))
        ax.set_title(f"Biomass distribution – KS = {ks:.3f}")
    else:
        ax.set_title("Biomass distribution")
        

    ax.set_xlabel("log\u2081\u2080(Biomass)");  ax.set_ylabel("Density")
    ax.legend(frameon=False);  fig.tight_layout()
    fig.savefig(filename, dpi=300);  plt.close(fig)
    logger.info("Saved figure %s", filename)


# ----------- NEW “visual helpers” ---------------------------------------------------
def density_histogram(arr1, arr2, labels, fname, threshold):
    """
    Over-laid *density* histogram (area=1) of log₁₀ biomass.

    Parameters
    ----------
    arr1, arr2 : 1-D biomass arrays (flattened)
    labels     : ("IBM", "ODE") or ("IBM", "PSD") …
    threshold  : biomass values < threshold are ignored
    """
    a = np.log10(arr1[arr1 >= threshold])
    b = np.log10(arr2[arr2 >= threshold])
    bins = _auto_bins(np.concatenate([a, b]), "FD")          # reuse helper
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(a, bins=bins, density=True, alpha=.5, label=labels[0])
    ax.hist(b, bins=bins, density=True, alpha=.5, label=labels[1])
    ax.set_xlabel("log$_{10}$ biomass");  ax.set_ylabel("density")
    ax.set_title("Density histogram")
    ax.legend(frameon=False);  fig.tight_layout()
    fig.savefig(fname, dpi=300);  plt.close(fig)
    logger.info("Saved %s", fname)

def richness_timeseries(traj, threshold):
    """Richness at each record & its mean."""
    pres = traj > threshold
    rich = pres.sum(1)
    return rich, rich.mean()

def plot_richness(t, rich_ibm, rich_ode, rich_psd, fname):
    THRESH    = 1e-7
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(t, rich_ibm, label="IBM")
    ax.plot(t, rich_ode, label="ODE")
    ax.plot(t, rich_psd, label="PSD")
    ax.set_xlabel("time");  ax.set_ylabel(f"species ≥ {THRESH:g}")
    ax.set_title("Species richness vs. time");  ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    logger.info("Saved figure %s", fname)
# ------------------------------------------------------------------------------------

def invasion_count(traj, threshold):
    """
    Count #times each species crosses upward through 'threshold'.
    Works for IBM & PSD; for PSD we call this S→D or P→D.
    """
    above = traj > threshold
    # upward crossings: 0→1 between successive rows
    up = (above[1:] & (~above[:-1])).sum()
    return int(up)
# ------------------------------------------------------------------------------------

# ─────────────────────────── density histogram (3 models) ──────────────
def density_histogram_3(x, y, z, labels, fname, thresh, bins=30):
    """
    Over-laid density histogram of log10 biomasses > thresh
    for three 1-D arrays *x, y, z* (IBM, ODE, PSD).

    labels = ("IBM","ODE","PSD")
    """
    import matplotlib.pyplot as plt, numpy as np, logging
    logger = logging.getLogger(__name__)

    data   = [d[d > thresh] for d in (x, y, z)]
    logs   = [np.log10(d) for d in data]

    fig, ax = plt.subplots(figsize=(6,3))
    colours = ["C0","C1","C2"]
    for L, l, c in zip(labels, logs, colours):
        ax.hist(l, bins=bins, density=True,
                histtype="stepfilled", alpha=0.25,
                edgecolor=c, facecolor=c, label=L, linewidth=1.2)

    ax.set_xlabel("log$_{10}$ biomass");  ax.set_ylabel("density")
    ax.set_title(f"Threshold = {thresh:g}")
    ax.legend(frameon=False);  fig.tight_layout()
    fig.savefig(fname, dpi=300);  plt.close(fig)
    logger.info("Saved figure %s", fname)

# ────────────────────────────── invasion counter ───────────────────────
def invasion_timeseries(traj, threshold):
    """
    Return an array *inv[t]* = number of species that crossed
    the threshold *upwards* between record t-1 and t.
    """
    pres = traj > threshold
    up   = ( pres[1:] & ~pres[:-1] )      # rising edge
    return up.sum(axis=1)                 # per-time-step counts
# ────────────────────────────── save model outputs ─────────────────────

# ============  EXTRA HELPERS  =================================================
# ============  SINGLE-RUN 3×1 HISTOGRAM  =====================================
# ────────── helper ──────────────────────────────────────────────────────
def histogram_panel(npz_path:str, out_png:str,
                    *, bins:int=30, burn:float=0.5, thresh:float=1e-6):
    """
    Plot three stacked density histograms of log10-biomass (ODE, IBM, PSD2).

    Parameters
    ----------
    npz_path : str   path to model_outputs.npz
    out_png  : str   output PNG file
    bins     : int   histogram bins
    burn     : float fraction of trajectory discarded as burn-in (0–1)
    thresh   : float biomass threshold below which data are ignored
    """
    keys   = ["ODE", "IBM", "PSD2"]
    titles = ["ODE", "IBM", "PSD"]

    def load(key):
        arr  = np.load(npz_path)[key]             # shape = (T, S)
        arr  = arr[int(len(arr)*burn):].ravel()   # burn-in & flatten
        arr  = arr[arr > thresh]
        return np.log10(arr)

    fig, axs = plt.subplots(3, 1, figsize=(4.0, 6.0), sharex=True)
    for ax, key, title in zip(axs, keys, titles):
        ax.hist(load(key), bins=bins, density=True,
                alpha=0.7, color="#4C72B0")
        ax.set_ylabel("density")
        ax.set_title(title, loc="left")
    axs[-1].set_xlabel("log$_{10}$ biomass")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"histogram panel saved → {out_png}")



def split_invasion_counts(traj:np.ndarray, thresh:float):
    """
    Return two vectors:
        S→D  counts  (crossing UP through *thresh*)
        D→P  counts  (crossing DOWN through *thresh*)
    Both are length (#records-1).
    """
    above   = traj > thresh
    up      = (above[1:] & ~above[:-1]).sum(1)         # 0→1 transitions
    down    = (~above[1:] & above[:-1]).sum(1)         # 1→0 transitions
    return up, down


def save_model_output(ibm_trajectory, #psd_trajectory,
                      psd2_trajectory, psd2_waiting,
                      psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob,
                      ode_trajectory):
    """
    Save all model outputs (including PSD diagnostics) to a .npz file.
    """
    np.savez(
        "model_outputs.npz",
        IBM=ibm_trajectory,
        # PSD=psd_trajectory,
        PSD2=psd2_trajectory,
        PSD2_waiting=psd2_waiting,
        PSD2_poisson_clock=psd2_poisson_clock,
        PSD2_growth_rate=psd2_growth_rate,
        PSD2_invasion_rate=psd2_invasion_rate,
        PSD2_est_prob=psd2_est_prob,
        ODE=ode_trajectory
    )
    logger.info("Saved model outputs to model_outputs.npz for advanced visualization later.")
