
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
from config import THRESHOLD, BODY_MASS
from utils import mav, bav, mean_se

logger = logging.getLogger(__name__)

def alpha_diversity(trajectory, threshold=THRESHOLD):
    """
    Count how many species have biomass > threshold at each time record.
    """
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

def plot_trajectory(trajectories, labels, title, filename):
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

def histogram_biomass(trajectories, labels, filename):
    """
    Overlaid histograms (log10 scale) for biomass distributions.
    """
    plt.figure(figsize=(10,6))
    for traj, label in zip(trajectories, labels):
        data = traj.flatten()
        data = data[data > 0]
        if len(data) == 0:
            continue
        plt.hist(np.log10(data), bins=50, alpha=0.4, label=label, density=True)
    plt.xlabel("log10(Biomass)")
    plt.ylabel("Density")
    plt.title("Biomass Distribution")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def covariance_matrix_plot(trajectory, title, filename):
    """
    Plot the covariance matrix computed from the latter half of the trajectory.
    """
    half = trajectory.shape[0] // 2
    portion = trajectory[half:, :]
    covmat = np.cov(portion.T)
    plt.figure(figsize=(6,5))
    plt.imshow(covmat, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Covariance')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved figure {filename}")

def save_model_output(ibm_trajectory, psd_trajectory, psd2_trajectory, psd2_waiting,
                      psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob,
                      ode_trajectory):
    """
    Save all model outputs (including PSD2 diagnostics) to a .npz file.
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
