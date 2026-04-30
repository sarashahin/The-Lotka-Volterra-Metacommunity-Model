#!/usr/bin/env python3
"""
compute_turnover_rates.py

Compute mean "colour-change" (dominant species turnover) rates for each model.

AXEL'S SUGGESTION:
  "I believe we can say 'accurately' only if we do some quantitative comparison
   (e.g., the mean number of cycles per time, averaged over the simulation plane.
   We could say briefly, e.g., 'With IBM, sites change colour at an average rate
   of x ± y and similarly u ± v for PSD. For ODE, this rate (s ± t) is much higher.'"

This script computes:
  - For each model, at each site (patch), count how many times the dominant
    species changes identity over the simulation.
  - Report mean ± SE of the turnover rate (changes per time unit) averaged
    over all patches.

Usage:
  python compute_turnover_rates.py --npz results/data/rps_dataset.npz

Output example:
  ODE:  turnover rate = 0.0234 ± 0.0012 per time unit
  IBM:  turnover rate = 0.0003 ± 0.0001 per time unit
  PSD:  turnover rate = 0.0004 ± 0.0001 per time unit

These values can then be inserted into the LaTeX text.
"""

import argparse
import numpy as np


def load_model(npz_path, model):
    """Load biomass array and time vector for a model."""
    d = np.load(npz_path)
    key_map = {
        "ODE": ["ODE_B", "ODE"],
        "IBM": ["IBM_B", "IBM"],
        "PSD": ["PSD2_B", "PSD2", "PSD_B", "PSD"],
    }
    t_map = {
        "ODE": ["ODE_t", "t"],
        "IBM": ["IBM_t", "t"],
        "PSD": ["PSD2_t", "PSD_t", "t"],
    }

    B = None
    for k in key_map.get(model, [model]):
        if k in d:
            B = np.asarray(d[k])
            break
    if B is None:
        raise KeyError(f"Cannot find {model} in {npz_path}")

    t = None
    for k in t_map.get(model, ["t"]):
        if k in d:
            t = np.asarray(d[k]).ravel()
            break
    if t is None:
        raise KeyError(f"Cannot find time vector for {model}")

    return B, t  # B: (T, S, Ny, Nx), t: (T,)


def compute_turnover_rate(B, t, burn_frac=0.3):
    """
    Compute per-site dominant-species turnover rate.

    Parameters
    ----------
    B : ndarray, shape (T, S, Ny, Nx)
        Biomass array
    t : ndarray, shape (T,)
        Time vector
    burn_frac : float
        Fraction of initial time series to discard as burn-in

    Returns
    -------
    mean_rate : float
        Mean turnover rate (changes per time unit) across all patches
    se_rate : float
        Standard error of the mean
    per_site_rates : ndarray, shape (Ny, Nx)
        Turnover rate at each patch
    """
    T, S, Ny, Nx = B.shape

    # Discard burn-in
    cut = int(burn_frac * T)
    B_ss = B[cut:]
    t_ss = t[cut:]
    duration = t_ss[-1] - t_ss[0]

    # Dominant species at each time and patch
    dom = B_ss.argmax(axis=1)  # (T_ss, Ny, Nx)

    # Count transitions: where dominant species changes from one step to next
    changes = (dom[1:] != dom[:-1])  # (T_ss-1, Ny, Nx)

    # Total changes per site
    n_changes = changes.sum(axis=0)  # (Ny, Nx)

    # Rate = changes / duration
    per_site_rates = n_changes / duration

    # Mean and SE across patches
    flat_rates = per_site_rates.ravel()
    mean_rate = np.mean(flat_rates)
    se_rate = np.std(flat_rates, ddof=1) / np.sqrt(len(flat_rates))

    return mean_rate, se_rate, per_site_rates


def main():
    ap = argparse.ArgumentParser(
        description="Compute dominant-species turnover rates for ODE, IBM, PSD"
    )
    ap.add_argument("--npz", required=True, help="Path to rps_dataset.npz")
    ap.add_argument("--burn", type=float, default=0.3,
                    help="Burn-in fraction (default: 0.3)")
    ap.add_argument("--models", default="ODE,IBM,PSD",
                    help="Comma-separated models (default: ODE,IBM,PSD)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    print("=" * 65)
    print("DOMINANT-SPECIES TURNOVER RATES")
    print(f"Burn-in: {args.burn:.0%}")
    print("=" * 65)

    results = {}
    for model in models:
        B, t = load_model(args.npz, model)
        mean_r, se_r, _ = compute_turnover_rate(B, t, burn_frac=args.burn)
        results[model] = (mean_r, se_r)

        # Format nicely with scientific notation if needed
        if mean_r < 0.01:
            print(f"  {model:>4s}:  turnover rate = ({mean_r:.2e}) ± ({se_r:.2e}) per time unit")
        else:
            print(f"  {model:>4s}:  turnover rate = {mean_r:.4f} ± {se_r:.4f} per time unit")

    print()
    print("-" * 65)
    print("LATEX SNIPPET (paste into Section 5.2 if IBM ≈ PSD ≪ ODE):")
    print("-" * 65)

    # Generate LaTeX snippet
    parts = []
    for model in models:
        mean_r, se_r = results[model]
        display = "PSD" if model == "PSD" else model
        # Format as scientific notation for LaTeX
        if mean_r < 0.01:
            exp = int(np.floor(np.log10(mean_r)))
            coeff = mean_r / (10 ** exp)
            se_coeff = se_r / (10 ** exp)
            parts.append(
                f"{display}: $({coeff:.2f} \\pm {se_coeff:.2f}) \\times 10^{{{exp}}}$"
            )
        else:
            parts.append(f"{display}: ${mean_r:.4f} \\pm {se_r:.4f}$")

    print()
    print("With IBM, sites change dominant species at an average rate of")
    print(f"  {parts[models.index('IBM')]}")
    print(f"per time unit, and similarly {parts[models.index('PSD')]} for PSD.")
    print(f"For ODE, this rate ({parts[models.index('ODE')]}) is much higher,")
    print("confirming that PSD accurately captures metacommunity-scale")
    print("stochastic dynamics at a fraction of the computational cost.")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()