#!/usr/bin/env python3
"""
make_hist_panel_v11.py

PRL-style 3x1 histograms with:
- Biomass-based state classification (D/S/P) using growth rate sign
- Shared X-axis AND Y-axis across all panels (same scale for direct comparison)
- Threshold 1e-9 (matches v4) to filter very low biomass
- Counts by default (use --use_density for density normalization)
- Optional overflow mode for PSD panel (--psd_overflow)
- NEW: --no_title flag to remove m₀ title above panels (for cleaner typography)

Key difference from v9: Added --no_title flag per Axel's feedback:
  "remove the m₀ values above the panels because you already have them below"

State classification:
  - D-state: B > 0.01 (established)
  - S-state: B ≤ 0.01 AND ĝ > 0 (waiting, can invade)
  - P-state: B ≤ 0.01 AND ĝ < 0 (excluded)

Usage:
  # Standard mode (no title above panels - recommended for publication):
  python make_hist_panel_v11.py \\
      --npz "body-mass 1e-4 --inv 1e-10 --S 500 --seeds456/model_outputs.npz" \\
      --out vis/hist_v11/fig2b_high_mass.pdf \\
      --body_mass 1e-4 \\
      --xmin -8 --xmax 0 \\
      --no_title

  # With title (for standalone figures):
  python make_hist_panel_v11.py \\
      --npz "body-mass 1e-11 --inv 1e-10 --S 500 --seeds456/model_outputs.npz" \\
      --out vis/hist_v11/fig2a_low_mass.pdf \\
      --body_mass 1e-11 \\
      --xmin -8 --xmax 0
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def resolve_one(path_or_glob: str) -> str:
    if os.path.exists(path_or_glob):
        return path_or_glob
    matches = glob.glob(path_or_glob)
    if not matches:
        raise SystemExit(f"No file matches: {path_or_glob}")
    matches.sort()
    return matches[0]


def load_data(path):
    """Load all arrays from NPZ file."""
    d = np.load(path)
    print(f"[info] Keys: {sorted(d.keys())}")
    
    O = d.get('ODE', d.get('ode'))
    I = d.get('IBM', d.get('ibm'))
    
    P = None
    for k in ['PSD2', 'PSD']:
        if k in d:
            P = np.asarray(d[k])
            break
    
    G = None
    for k in ['PSD2_growth_rate', 'PSD_growth_rate', 'growth_rate']:
        if k in d:
            G = np.asarray(d[k])
            break
    
    return np.asarray(O), np.asarray(I), P, G


def classify_states(P, G, body_mass, d_thresh=1e-2):
    """
    Body-mass-dependent state classification.
    
    For HIGH body-mass (m₀ ≥ 1e-6): Use threshold B > d_thresh for D-state
    For LOW body-mass (m₀ < 1e-6): Use only sign of ĝ (no threshold needed)
    
    This follows Eq. (6) in the manuscript: at low m₀, transitions into 
    S-state are rare, so species flip between P (ĝ < 0) and D (ĝ > 0).
    """
    states = np.zeros_like(P, dtype=int)  # Default: P-state (0)
    
    if body_mass < 1e-6:
        # LOW body-mass regime: classify by growth rate sign only
        # No biomass threshold needed - species flip between P and D
        states[G > 0] = 2   # D-state (positive growth, can persist)
        states[G <= 0] = 0  # P-state (negative growth, excluded)
        # S-state (1) essentially never assigned - transitions are rare
    else:
        # HIGH body-mass regime: use biomass threshold
        states[P > d_thresh] = 2                    # D-state (established)
        states[(P <= d_thresh) & (G > 0)] = 1       # S-state (waiting)
        # P-state (0) for B ≤ threshold AND ĝ ≤ 0
    
    return states


def stack_after_burn(X, burn_frac, thresh, xmode):
    cut = int(burn_frac * X.shape[0])
    flat = X[cut:].ravel()
    flat = flat[flat > thresh]
    if len(flat) == 0:
        return np.array([])
    return np.log10(flat) if xmode == "log" else flat


def stack_with_states(X, states, burn_frac, thresh, xmode):
    """Stack data after burn-in, separating by state."""
    cut = int(burn_frac * X.shape[0])
    X_post = X[cut:].ravel()
    states_post = states[cut:].ravel()
    mask = X_post > thresh
    
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    to_log = lambda x: np.log10(x) if xmode == "log" else x
    
    all_data = to_log(X_post[mask])
    d_data = to_log(X_post[mask & (states_post == 2)])
    s_data = to_log(X_post[mask & (states_post == 1)])
    p_data = to_log(X_post[mask & (states_post == 0)])
    
    return all_data, d_data, s_data, p_data


def main():
    ap = argparse.ArgumentParser(description="Histograms with shared y-axes (v11 with --no_title)")
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xmode", choices=["log", "linear"], default="log")
    ap.add_argument("--burn", type=float, default=0.7)
    ap.add_argument("--thresh", type=float, default=1e-9)
    ap.add_argument("--nbins", type=int, default=25)
    ap.add_argument("--body_mass", type=float, default=1e-4)
    ap.add_argument("--xmin", type=float, default=-8.0)
    ap.add_argument("--xmax", type=float, default=0.0)
    ap.add_argument("--d_thresh", type=float, default=1e-2)
    ap.add_argument("--use_density", action="store_true",
                    help="Use density normalization instead of counts (default: counts)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="Manual y-axis maximum")
    ap.add_argument("--psd_overflow", action="store_true",
                    help="Allow PSD panel bars to overflow beyond y-axis (for high body-mass)")
    ap.add_argument("--no_title", action="store_true",
                    help="Remove m₀ title above panels (recommended for subfigures in LaTeX)")
    
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load data
    npz_path = resolve_one(args.npz)
    print(f"\n{'='*60}")
    print(f"LOADING: {npz_path}")
    print(f"{'='*60}")
    
    O_raw, I_raw, P_raw, G_raw = load_data(npz_path)
    
    # Clean arrays
    O_raw = np.nan_to_num(O_raw, nan=0.0)
    I_raw = np.nan_to_num(I_raw, nan=0.0)
    P_raw = np.nan_to_num(P_raw, nan=0.0)
    O_raw[O_raw < 0] = 0
    I_raw[I_raw < 0] = 0
    P_raw[P_raw < 0] = 0

    # State classification
    if G_raw is None:
        raise SystemExit("ERROR: Growth rate data not found!")
    
    states = classify_states(P_raw, G_raw, args.body_mass, d_thresh=args.d_thresh)
    
    # Process data
    O = stack_after_burn(O_raw, args.burn, args.thresh, args.xmode)
    I = stack_after_burn(I_raw, args.burn, args.thresh, args.xmode)
    P_all, P_d, P_s, P_p = stack_with_states(P_raw, states, args.burn, args.thresh, args.xmode)
    
    print(f"\n[SAMPLE COUNTS]")
    print(f"    ODE: {len(O):,}")
    print(f"    IBM: {len(I):,}")
    print(f"    PSD: {len(P_all):,} (D:{len(P_d):,}, S:{len(P_s):,}, P:{len(P_p):,})")
    
    if len(P_s) + len(P_p) > 0:
        s_frac = 100 * len(P_s) / (len(P_s) + len(P_p))
        p_frac = 100 * len(P_p) / (len(P_s) + len(P_p))
        print(f"    Non-D breakdown: S={s_frac:.1f}%, P={p_frac:.1f}%")

    # ==================== CREATE FIGURE ====================
    bins = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    
    use_density = args.use_density
    ylabel = "density" if use_density else "count"
    
    # Create figure - adjust height based on whether title is shown
    fig_height = 7.0 if args.no_title else 7.2
    fig, axs = plt.subplots(3, 1, figsize=(5.2, fig_height), sharex=True, sharey=True)
    
    # Colors
    col_ode = "#8fb0ff"
    col_ibm = "#99d39b"
    col_d = "#d69a9a"
    col_s = "#ffb366"
    col_p = "#b366ff"

    # ----- ODE panel -----
    ax = axs[0]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(O) > 0:
        counts_ode, _, patches = ax.hist(O, bins=bins, density=use_density, 
                facecolor=col_ode, edgecolor="0.25", 
                linewidth=0.8, alpha=0.65)
    ax.set_ylabel(ylabel)
    ax.set_title("ODE", loc="left", fontweight="bold")

    # ----- IBM panel -----
    ax = axs[1]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(I) > 0:
        _, _, patches = ax.hist(I, bins=bins, density=use_density,
                facecolor=col_ibm, edgecolor="0.25",
                linewidth=0.8, alpha=0.65)
    else:
        ax.text(0.5, 0.5, "IBM data sparse", transform=ax.transAxes,
               ha='center', va='center', fontsize=10, color='gray', style='italic')
    ax.set_ylabel(ylabel)
    ax.set_title("IBM", loc="left", fontweight="bold")

    # ----- PSD panel with stacked states -----
    ax = axs[2]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    data_to_plot = []
    colors_to_plot = []
    legend_elements = []
    
    if len(P_p) > 0:
        data_to_plot.append(P_p)
        colors_to_plot.append(col_p)
        legend_elements.append(Patch(facecolor=col_p, edgecolor='0.25', 
                                    alpha=0.75, label='P (excluded, ĝ<0)'))
    
    if len(P_s) > 0:
        data_to_plot.append(P_s)
        colors_to_plot.append(col_s)
        legend_elements.append(Patch(facecolor=col_s, edgecolor='0.25', 
                                    alpha=0.75, label='S (waiting, ĝ>0)'))
    
    if len(P_d) > 0:
        data_to_plot.append(P_d)
        colors_to_plot.append(col_d)
        legend_elements.append(Patch(facecolor=col_d, edgecolor='0.25', 
                                    alpha=0.75, label='D (established)'))
    
    if data_to_plot:
        if use_density:
            total_samples = len(P_all)
            bin_width = bins[1] - bins[0]
            
            bottom = np.zeros(len(bins) - 1)
            for data, color in zip(data_to_plot, colors_to_plot):
                counts, _ = np.histogram(data, bins=bins)
                density = counts / (total_samples * bin_width)
                bars = ax.bar(bins[:-1], density, width=bin_width, bottom=bottom,
                      color=color, edgecolor="0.25", linewidth=0.5, alpha=0.75, align='edge')
                if args.psd_overflow:
                    for bar in bars:
                        bar.set_clip_on(False)
                bottom += density
        else:
            _, _, patches_list = ax.hist(data_to_plot, bins=bins, stacked=True,
                    color=colors_to_plot, edgecolor="0.25", linewidth=0.5, alpha=0.75)
            
            if args.psd_overflow:
                for patches in patches_list:
                    for patch in patches:
                        patch.set_clip_on(False)
        
        ax.legend(handles=legend_elements[::-1], loc='upper right', fontsize=7, framealpha=0.9)
    
    ax.set_ylabel(ylabel)
    ax.set_title("PSD", loc="left", fontweight="bold")
    
    # Set axis limits
    for ax in axs:
        ax.set_xlim(args.xmin, args.xmax)
    
    if args.ymax:
        for ax in axs:
            ax.set_ylim(0, args.ymax)
    
    # X-axis label
    axs[-1].set_xlabel(r"$\log_{10}$ biomass" if args.xmode == "log" else "Biomass")
    
    # Title - only show if --no_title is NOT set
    if not args.no_title:
        m0_exp = int(np.log10(args.body_mass))
        fig.suptitle(f"$m_0 = 10^{{{m0_exp}}}$", fontsize=11, y=0.98)
    
    # Adjust layout
    if args.psd_overflow:
        fig.subplots_adjust(left=0.15, right=0.95, top=0.97 if args.no_title else 0.94, 
                           bottom=0.08, hspace=0.25)
    else:
        if args.no_title:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {args.out}")
    
    if args.out.endswith('.pdf'):
        png_path = args.out.replace('.pdf', '.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {png_path}")
    
    plt.close(fig)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Body-mass: {args.body_mass}")
    print(f"  Title above panels: {'NO (--no_title)' if args.no_title else 'YES'}")
    print(f"  X-axis: [{args.xmin}, {args.xmax}]")
    if args.ymax:
        print(f"  Y-max: {args.ymax}")
    if args.psd_overflow:
        print(f"  PSD overflow: ENABLED")
    print(f"  Mode: {'density' if use_density else 'counts'}")
    if len(P_s) + len(P_p) > 0:
        print(f"  S/P ratio: {s_frac:.0f}% / {p_frac:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()