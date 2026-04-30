
#!/usr/bin/env python3
"""
make_mosaic_movie_panels.py

Publication-quality spatial mosaic panels for RPS metacommunity dynamics.

SUPERVISOR FEEDBACK V5:
  - Add coarse-then-fine time selection: first frames widely spaced,
    last 3 frames closely spaced to show gradual changes
  - "Do the same as here, but then maybe for the last three steps 1,2,3
     have the time difference much smaller so we can actually see how
     these things slowly change."

Usage Examples:
  # Coarse-fine times: 6 coarse frames + 3 fine frames at the end
  python make_mosaic_movie_panels.py --npz results/data/rps_dataset.npz \\
      --models ODE,IBM,PSD2 \\
      --coarse-fine-times 6,3 --fine-start 80000 --fine-end 90000 \\
      --out figures/fig_rps_coarse_fine.pdf --format pdf

  # Alternative: specify exact times with fine spacing at end
  python make_mosaic_movie_panels.py --npz results/data/rps_dataset.npz \\
      --models ODE,IBM,PSD2 \\
      --times 0,20000,40000,60000,80000,85000,87500,90000 \\
      --out figures/fig_rps_movie_fine.pdf --format pdf
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ------------------------- loading helpers ------------------------------

def _get(arrdict, *names):
    """Get array from dict, trying multiple key names."""
    for n in names:
        if n in arrdict:
            return arrdict[n]
    raise KeyError(f"None of {names} found in NPZ")


def load_model(npz_path: str, model: str):
    """Load biomass array and time vector for specified model."""
    d = np.load(npz_path)
    model = model.upper()
    
    if model == "IBM":
        B = _get(d, "IBM_B", "IBM")
        t = _get(d, "IBM_t", "t")
    elif model in ("PSD2", "PSD"):
        B = _get(d, "PSD2_B", "PSD2", "PSD_B", "PSD")
        t = _get(d, "PSD2_t", "PSD_t", "t")
    elif model == "ODE":
        B = _get(d, "ODE_B", "ODE")
        t = _get(d, "ODE_t", "t")
    else:
        raise SystemExit(f"Unknown model {model}")
    
    B = np.asarray(B)  # shape (T, S, Ny, Nx)
    t = np.asarray(t).ravel()
    
    if B.ndim != 4:
        raise SystemExit(f"B must be (T, S, Ny, Nx), got {B.shape}")
    if B.shape[0] != t.shape[0]:
        raise SystemExit(f"T mismatch: B has {B.shape[0]} frames, t has {t.shape[0]}")
    
    return B, t


# ---------------------- stable palette / legend -------------------------

def pick_global_topk(B: np.ndarray, top_k: int):
    """
    Select top-K species by total biomass across time and space.
    Returns sorted species indices and their totals.
    """
    totals = B.sum(axis=(0, 2, 3))  # (S,)
    order = np.argsort(totals)[::-1]
    keep = order[:min(top_k, B.shape[1])]
    keep = np.sort(keep)  # ascending order for consistency
    return keep, totals


def pick_global_topk_multi(B_list: list, top_k: int):
    """
    Select top-K species consistently across multiple model outputs.
    Combines totals from all models for stable ranking.
    """
    combined = None
    for B in B_list:
        totals = B.sum(axis=(0, 2, 3))
        if combined is None:
            combined = totals.copy()
        else:
            combined += totals
    
    order = np.argsort(combined)[::-1]
    keep = order[:min(top_k, len(combined))]
    keep = np.sort(keep)
    return keep, combined


def journal_palette(K: int):
    """
    Publication-quality colour palette.
    Bright, distinct colours matching the original (red, blue, green).
    """
    # Match the original palette: red, blue, green
    base = np.array([
        [0.88, 0.15, 0.07],  # red (Species 1)
        [0.12, 0.47, 0.71],  # blue (Species 2)
        [0.17, 0.63, 0.17],  # green (Species 3)
        [0.84, 0.37, 0.00],  # orange
        [0.58, 0.40, 0.74],  # purple
        [0.55, 0.34, 0.29],  # brown
        [0.89, 0.47, 0.76],  # pink
        [0.09, 0.75, 0.81],  # cyan
        [0.74, 0.74, 0.13],  # olive
        [0.55, 0.55, 0.55],  # grey
    ])
    
    if K <= len(base):
        return base[:K]
    
    reps = int(np.ceil(K / len(base)))
    extended = np.vstack([base for _ in range(reps)])[:K]
    return extended


def slice_to_rgb_fixedpalette(state: np.ndarray,
                              keep_ids: np.ndarray,
                              colour_table: np.ndarray,
                              empty_colour=(1.0, 1.0, 1.0),  # white for journal
                              ) -> np.ndarray:
    """
    Convert biomass state to RGB image with fixed species-colour mapping.
    
    Note: No "Other species" category - Axel confirmed there are only 3 species
    in the RPS model, plus empty patches.
    """
    S, Ny, Nx = state.shape
    idx_dom = state.argmax(axis=0)  # dominant species per pixel
    occ = state.sum(axis=0) > 0     # occupied pixels
    
    # Start with empty colour (white for journal style)
    img = np.zeros((Ny, Nx, 3), float)
    img[:] = empty_colour
    
    # Map species ID -> palette slot
    inv = {int(s): i for i, s in enumerate(keep_ids)}
    
    for s in keep_ids:
        mask = (idx_dom == s) & occ
        img[mask] = colour_table[inv[int(s)]]
    
    return img


# -------------------------- time selection ------------------------------

def indices_for_times(t, wanted, allow_duplicates=False):
    """Find frame indices closest to specified times.
    
    Parameters
    ----------
    t : array
        Time vector
    wanted : array
        Desired time values
    allow_duplicates : bool
        If True, keep duplicate indices (for consistent frame counts across models)
    """
    wanted = np.asarray(wanted, float)
    idx = [int(np.argmin(np.abs(t - w))) for w in wanted]
    
    if allow_duplicates:
        return idx
    
    # Enforce strict increase & uniqueness
    uniq = []
    last = -1
    for i in idx:
        if i != last:
            uniq.append(i)
            last = i
    return uniq


def auto_pick_times_movie(B, t, nframes=8, target_change=0.05):
    """
    Pick frames with small incremental changes for movie-like continuity.
    
    Supervisor feedback: "shorter time steps so I can see how shapes change"
    """
    T, S, Ny, Nx = B.shape
    chosen = [0]
    dom = B.argmax(axis=1)  # (T, Ny, Nx)
    
    while len(chosen) < nframes:
        last = chosen[-1]
        best_j, best_diff = None, 1e9
        
        for j in range(last + 1, T):
            diff = (dom[last] != dom[j]).mean()
            score = abs(diff - target_change)
            if score < best_diff:
                best_diff, best_j = score, j
        
        if best_j is None or best_j == last:
            # If we can't find good frames, space evenly
            remaining = nframes - len(chosen)
            if remaining > 0 and T > chosen[-1] + 1:
                step = max(1, (T - 1 - chosen[-1]) // (remaining + 1))
                for k in range(1, remaining + 1):
                    next_idx = min(chosen[-1] + k * step, T - 1)
                    if next_idx > chosen[-1]:
                        chosen.append(next_idx)
            break
        
        chosen.append(best_j)
    
    return chosen


def auto_pick_times_even(t, nframes=8):
    """Pick evenly spaced frames across the time range."""
    T = len(t)
    if nframes >= T:
        return list(range(T))
    
    indices = np.linspace(0, T - 1, nframes, dtype=int)
    return list(np.unique(indices))


def generate_coarse_fine_times(t, n_coarse, n_fine, fine_start, fine_end):
    """
    Generate time points with coarse spacing initially, then fine spacing at the end.
    
    SUPERVISOR FEEDBACK:
    "Do the same as here, but then maybe for the last three steps 1,2,3 
     have the time difference much smaller so we can actually see how 
     these things slowly change."
    
    Parameters
    ----------
    t : array
        Time vector from simulation
    n_coarse : int
        Number of coarsely-spaced frames (covering t[0] to fine_start)
    n_fine : int
        Number of finely-spaced frames (covering fine_start to fine_end)
    fine_start : float
        Time at which fine sampling begins
    fine_end : float
        Time at which fine sampling ends (usually end of simulation)
    
    Returns
    -------
    times : list
        List of time values combining coarse and fine sampling
    
    Example
    -------
    For n_coarse=6, n_fine=3, fine_start=80000, fine_end=90000:
    
    Coarse part: t=0 to t=80000 in 6 steps
        → [0, 16000, 32000, 48000, 64000, 80000]
    
    Fine part: t=80000 to t=90000 in 3 steps (excluding 80000 which is already included)
        → [83333, 86667, 90000]
    
    Combined: [0, 16000, 32000, 48000, 64000, 80000, 83333, 86667, 90000]
    """
    t_min = t[0]
    t_max = t[-1]
    
    # Validate inputs
    if fine_start >= fine_end:
        raise ValueError(f"fine_start ({fine_start}) must be < fine_end ({fine_end})")
    if fine_start < t_min:
        fine_start = t_min
    if fine_end > t_max:
        fine_end = t_max
    
    # Generate coarse times: from t_min to fine_start (inclusive)
    if n_coarse > 1:
        coarse_times = np.linspace(t_min, fine_start, n_coarse).tolist()
    else:
        coarse_times = [t_min]
    
    # Generate fine times: from fine_start to fine_end
    # Exclude fine_start since it's already in coarse_times
    if n_fine > 0:
        fine_times_full = np.linspace(fine_start, fine_end, n_fine + 1)
        fine_times = fine_times_full[1:].tolist()  # exclude first (already in coarse)
    else:
        fine_times = []
    
    # Combine
    all_times = coarse_times + fine_times
    
    return all_times


# ------------------------------ plotting -------------------------------

def add_legend_journal(fig, keep_ids, colours, ncol=5, fontsize=16):
    """
    Add publication-quality legend at bottom of figure.
    
    Species are labelled 1, 2, 3 (not 0, 1, 2) per supervisor feedback.
    No "Other species" - only the 3 RPS species + Empty.
    """
    handles = []
    labels = []
    
    for s, c in zip(keep_ids, colours):
        patch = Patch(facecolor=c, edgecolor='black', linewidth=0.5)
        handles.append(patch)
        # Species numbered from 1, not 0 (supervisor feedback)
        labels.append(f"Species {int(s) + 1}")
    
    # Add "Empty" entry (white)
    handles.append(Patch(facecolor='white', edgecolor='black', linewidth=0.5))
    labels.append("Empty")
    
    leg = fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(ncol, len(labels)),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=fontsize,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.4,
    )
    
    return leg


def make_single_model_panel(B, t, keep_ids, colours, idx_list, out_path,
                            ncols=8, dpi=300, suptitle=None, figformat='png'):
    """
    Create single-model mosaic panel (original style, updated for journal).
    """
    frames = B[idx_list]
    times = t[idx_list]
    n = len(idx_list)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(1.8 * ncols, 1.8 * nrows + 0.8),
        squeeze=False
    )
    
    # White background for journal (supervisor feedback)
    fig.patch.set_facecolor('white')
    
    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('white')
    
    for k, (snap, tt) in enumerate(zip(frames, times)):
        ax = axes.flat[k]
        img = slice_to_rgb_fixedpalette(snap, keep_ids, colours)
        ax.imshow(img, interpolation='nearest')
        # Black text on white background (supervisor feedback)
        ax.set_title(f"t = {tt:,.0f}", color='black', fontsize=10, pad=3)
        # Add thin border around each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
    
    if suptitle:
        fig.suptitle(suptitle, color='black', fontsize=12, fontweight='bold', y=0.98)
    
    add_legend_journal(fig, keep_ids, colours)
    
    plt.tight_layout(rect=(0.01, 0.06, 0.99, 0.95))
    
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    # Save in requested format
    if figformat.lower() == 'pdf':
        out_path = out_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(out_path, dpi=dpi, format='pdf', facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
    elif figformat.lower() == 'svg':
        out_path = out_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(out_path, dpi=dpi, format='svg', facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
    else:
        plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
    
    plt.close()
    print(f"[save] → {out_path}")
    return out_path


def make_multi_model_panel(data_dict, keep_ids, colours, idx_dict, out_path,
                           ncols=8, dpi=300, figformat='png'):
    """
    Create multi-model stacked panel figure.
    
    Supervisor feedback: "Put as Panel A, Panel B... one on top of each other"
    
    Parameters
    ----------
    data_dict : dict
        {model_name: (B, t)} for each model to include
    keep_ids : ndarray
        Shared species indices for consistent colouring
    colours : ndarray
        Colour palette
    idx_dict : dict
        {model_name: [frame_indices]} - can have different indices per model
    out_path : str
        Output file path
    """
    model_names = list(data_dict.keys())
    n_models = len(model_names)
    
    # Get number of time points (should be same for all models now)
    n_times = len(idx_dict[model_names[0]])
    
    # Verify all models have the same number of frames
    for model in model_names:
        if len(idx_dict[model]) != n_times:
            raise ValueError(f"Model {model} has {len(idx_dict[model])} frames, expected {n_times}")
    
    # Figure sizing for journal - wider panels
    panel_width = 1.6
    panel_height = 1.6
    label_width = 0.6
    fig_width = panel_width * n_times + label_width + 0.5
    fig_height = panel_height * n_models + 1.6  # extra for legend
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    # Create grid: rows for models, columns for time points + label column
    gs = gridspec.GridSpec(
        n_models, n_times + 1,
        width_ratios=[0.18] + [1] * n_times,
        hspace=0.15,
        wspace=0.05,
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.18
    )
    
    panel_labels = 'ABCDEFGHIJ'
    
    for row, model_name in enumerate(model_names):
        B, t = data_dict[model_name]
        idx_list = idx_dict[model_name]
        
        # Panel label (A, B, C...) with model name
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis('off')
        
        # Display name: PSD instead of PSD2 per original code
        display_name = "PSD" if model_name.upper() in {"PSD2", "PSD"} else model_name.upper()
        label_text = f"({panel_labels[row]}) {display_name}"
        ax_label.text(0.9, 0.5, label_text,
                            ha='right', va='center', fontsize=18, fontweight='bold',
                            transform=ax_label.transAxes, rotation=0)
        
        # Plot each time point
        for col, idx in enumerate(idx_list):
            ax = fig.add_subplot(gs[row, col + 1])
            
            snap = B[idx]
            img = slice_to_rgb_fixedpalette(snap, keep_ids, colours)
            ax.imshow(img, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Time labels only on top row
            if row == 0:
                            ax.set_title(f"t = {t[idx]:,.0f}", fontsize=16, color='black', pad=6)
            
            # Add thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)
    
    # Legend at bottom
    add_legend_journal(fig, keep_ids, colours, ncol=len(keep_ids) + 1, fontsize=16
                       )
    
    # Save
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    if figformat.lower() == 'pdf':
        out_path = out_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(out_path, dpi=dpi, format='pdf', facecolor='white',
                    bbox_inches='tight', pad_inches=0.1)
    elif figformat.lower() == 'svg':
        out_path = out_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(out_path, dpi=dpi, format='svg', facecolor='white',
                    bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig(out_path, dpi=dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=0.1)
    
    plt.close()
    print(f"[save] → {out_path}")
    return out_path


# ------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate publication-quality mosaic panels for Nature journal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Coarse-then-fine time selection (supervisor feedback)
  # 6 coarse frames from t=0 to t=80000, then 3 fine frames from t=80000 to t=90000
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --models ODE,IBM,PSD2 \\
      --coarse-fine-times 6,3 --fine-start 80000 --fine-end 90000 \\
      --out figures/fig_rps_coarse_fine.pdf --format pdf

  # Specific times (manually specify the coarse+fine pattern)
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --models ODE,IBM,PSD2 \\
      --times 0,20000,40000,60000,80000,85000,87500,90000 \\
      --out figures/fig_rps_movie_fine.pdf --format pdf

  # Single model with auto-selected times (movie effect)
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --model IBM \\
      --auto-times 8 --out figures/ibm_mosaic.png

  # Multiple models stacked as Panel A, B, C (recommended for Nature)
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --models ODE,IBM,PSD2 \\
      --auto-times 8 --out figures/fig_rps_dynamics.pdf --format pdf

  # Evenly spaced times
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --models ODE,IBM,PSD2 \\
      --even-times 8 --out figures/mosaic.pdf --format pdf

  # High resolution for print (600 DPI)
  python make_mosaic_movie_panels.py --npz rps_dataset.npz --models ODE,IBM,PSD2 \\
      --coarse-fine-times 6,3 --fine-start 80000 --fine-end 90000 \\
      --out figures/fig2.pdf --format pdf --dpi 600
        """
    )
    
    ap.add_argument("--npz", required=True,
                    help="Path to rps_dataset.npz")
    
    # Model selection (single or multiple)
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=["IBM", "PSD2", "PSD", "ODE"],
                             help="Single model to plot")
    model_group.add_argument("--models",
                             help="Comma-separated models for stacked panels (e.g., ODE,IBM,PSD2)")
    
    # Time selection
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument("--times", type=str, default=None,
                            help="Comma-separated times (e.g., 0,20000,40000,...)")
    time_group.add_argument("--auto-times", type=int, default=None,
                            help="Auto-pick N frames with movie-like continuity")
    time_group.add_argument("--even-times", type=int, default=None,
                            help="Pick N evenly spaced frames")
    time_group.add_argument("--coarse-fine-times", type=str, default=None,
                            help="Coarse,Fine counts (e.g., '6,3' for 6 coarse + 3 fine frames). "
                                 "Use with --fine-start and --fine-end.")
    
    # Coarse-fine parameters
    ap.add_argument("--fine-start", type=float, default=None,
                    help="Time at which fine sampling begins (for --coarse-fine-times)")
    ap.add_argument("--fine-end", type=float, default=None,
                    help="Time at which fine sampling ends (for --coarse-fine-times)")
    
    ap.add_argument("--target-change", type=float, default=0.05,
                    help="Fraction of pixels changing between auto frames (default: 0.05)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of species to show (default: 3 for RPS)")
    ap.add_argument("--out", required=True,
                    help="Output file path")
    ap.add_argument("--ncols", type=int, default=9,
                    help="Number of columns (time points) per row")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Resolution (use 300+ for print, 600 for high-quality)")
    ap.add_argument("--format", choices=['png', 'pdf', 'svg'], default='png',
                    help="Output format (pdf/svg recommended for journal)")
    
    args = ap.parse_args()
    
    # Determine which models to load
    if args.model:
        model_list = [args.model]
    else:
        model_list = [m.strip() for m in args.models.split(",")]
    
    # Load all models
    data_dict = {}
    for model in model_list:
        B, t = load_model(args.npz, model)
        data_dict[model] = (B, t)
        print(f"Loaded {model}: shape={B.shape}, t_range=[{t[0]:.0f}, {t[-1]:.0f}], {len(t)} frames")
    
    # Get consistent species ranking across all models
    B_list = [data_dict[m][0] for m in model_list]
    keep_ids, _ = pick_global_topk_multi(B_list, args.top_k)
    colours = journal_palette(len(keep_ids))
    
    # Determine frame indices for each model
    # First, get reference times from first model
    ref_model = model_list[0]
    B_ref, t_ref = data_dict[ref_model]
    
    # ==================== TIME SELECTION ====================
    if args.times:
        # Manual times
        wanted_times = [float(x) for x in args.times.split(",")]
        print(f"\n[time selection] Manual times specified")
        
    elif args.coarse_fine_times:
        # NEW: Coarse-then-fine time selection (supervisor feedback)
        parts = args.coarse_fine_times.split(",")
        if len(parts) != 2:
            raise SystemExit("--coarse-fine-times must be in format 'N_coarse,N_fine' (e.g., '6,3')")
        
        n_coarse = int(parts[0])
        n_fine = int(parts[1])
        
        # Determine fine range
        t_max = t_ref[-1]
        fine_start = args.fine_start if args.fine_start is not None else t_max * 0.8
        fine_end = args.fine_end if args.fine_end is not None else t_max
        
        print(f"\n[time selection] Coarse-then-fine (supervisor feedback)")
        print(f"    Coarse: {n_coarse} frames from t={t_ref[0]:.0f} to t={fine_start:.0f}")
        print(f"    Fine:   {n_fine} frames from t={fine_start:.0f} to t={fine_end:.0f}")
        
        wanted_times = generate_coarse_fine_times(t_ref, n_coarse, n_fine, fine_start, fine_end)
        
        # Print the time intervals to show the coarse vs fine difference
        intervals = [wanted_times[i+1] - wanted_times[i] for i in range(len(wanted_times)-1)]
        print(f"\n    Time intervals (Δt):")
        for i, dt in enumerate(intervals):
            label = "coarse" if i < n_coarse - 1 else "FINE"
            print(f"        {wanted_times[i]:.0f} → {wanted_times[i+1]:.0f}: Δt = {dt:.0f} ({label})")
        
    elif args.auto_times:
        ref_idx = auto_pick_times_movie(B_ref, t_ref, nframes=args.auto_times,
                                        target_change=args.target_change)
        wanted_times = [t_ref[i] for i in ref_idx]
        print(f"\n[time selection] Auto-movie ({args.auto_times} frames)")
        
    elif args.even_times:
        ref_idx = auto_pick_times_even(t_ref, nframes=args.even_times)
        wanted_times = [t_ref[i] for i in ref_idx]
        print(f"\n[time selection] Even spacing ({args.even_times} frames)")
        
    else:
        # Default: 8 frames with movie effect
        ref_idx = auto_pick_times_movie(B_ref, t_ref, nframes=8, target_change=0.05)
        wanted_times = [t_ref[i] for i in ref_idx]
        print(f"\n[time selection] Default auto-movie (8 frames)")
    
    print(f"\nTarget times: {[f'{t:.0f}' for t in wanted_times]}")
    print(f"Total frames: {len(wanted_times)}")
    
    # Now map these times to all models (ensures same number of frames)
    idx_dict = {}
    for model in model_list:
        B, t = data_dict[model]
        # Use allow_duplicates=True to ensure we get exactly len(wanted_times) frames
        idx = indices_for_times(t, wanted_times, allow_duplicates=True)
        
        idx_dict[model] = idx
        actual_times = [t[i] for i in idx]
        print(f"\n{model}:")
        print(f"    Frame indices: {idx}")
        print(f"    Actual times:  {[f'{tt:.0f}' for tt in actual_times]}")
    
    print(f"\nTop-{args.top_k} species (0-indexed): {keep_ids}")
    print(f"Species labels (1-indexed): {[s+1 for s in keep_ids]}")
    
    # Generate figure
    if len(model_list) == 1:
        # Single model: original style
        model = model_list[0]
        B, t = data_dict[model]
        display_name = "PSD" if model.upper() in {"PSD2", "PSD"} else model.upper()
        title = f"{display_name}: Dominant species mosaics"
        
        make_single_model_panel(
            B, t,
            keep_ids, colours, idx_dict[model], args.out,
            ncols=args.ncols, dpi=args.dpi,
            suptitle=title, figformat=args.format
        )
    else:
        # Multiple models: stacked panels A, B, C
        make_multi_model_panel(
            data_dict, keep_ids, colours, idx_dict, args.out,
            ncols=args.ncols, dpi=args.dpi, figformat=args.format
        )
    
    # Final summary
    print("\n" + "="*60)
    print("FIGURE GENERATED SUCCESSFULLY")
    if args.coarse_fine_times:
        print(f"  • Time selection: COARSE-THEN-FINE (supervisor feedback)")
        print(f"  • Last {n_fine} frames have SMALLER time intervals")
        print(f"  • This shows gradual changes in spatial patterns")
    print("="*60)


if __name__ == "__main__":
    main()
