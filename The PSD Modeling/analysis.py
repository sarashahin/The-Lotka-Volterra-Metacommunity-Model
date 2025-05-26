
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


def save_model_output(ibm_trajectory, psd_trajectory, psd2_trajectory, psd2_waiting,
                      psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob,
                      ode_trajectory):
    """
    Save all model outputs (including PSD diagnostics) to a .npz file.
    """
    np.savez(
        "model_outputs.npz",
        IBM=ibm_trajectory,
        PSD=psd_trajectory,
        PSD2=psd2_trajectory,
        PSD2_waiting=psd2_waiting,
        PSD2_poisson_clock=psd2_poisson_clock,
        PSD2_growth_rate=psd2_growth_rate,
        PSD2_invasion_rate=psd2_invasion_rate,
        PSD2_est_prob=psd2_est_prob,
        ODE=ode_trajectory
    )
    logger.info("Saved model outputs to model_outputs.npz for advanced visualization later.")
