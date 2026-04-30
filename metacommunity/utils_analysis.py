# utils_analysis.py
import numpy as np
from typing import Tuple

def count_invasions(traj: np.ndarray,
                    thresh: float = 1e-12
                   ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Count invasion events in a multi‑patch trajectory.

    Parameters
    ----------
    traj   : ndarray  (T, S, Ny, Nx)
             biomass time‑series returned by PSD2Model / IBMModel
    thresh : float
             presence threshold – must match the model's "THRESHOLD".

    Returns
    -------
    n_total         : int
        total # of False→True transitions across *all* species & patches
    invasions_sp_px : ndarray  (S, Ny, Nx)
        how often species *s* invaded patch (i,j)
    richness_px     : ndarray  (Ny, Nx)
        how many *different* species ever invaded patch (i,j)
    """
    # presence(t)  boolean mask
    present = traj > thresh                     # shape (T,S,Ny,Nx)
    # events(t)    True where it was absent at t‑1 and present at t
    ev      = present[1:] & ~present[:-1]       # shape (T‑1,S,Ny,Nx)

    invasions_sp_px = ev.sum(axis=0)            # (S,Ny,Nx)
    richness_px     = (invasions_sp_px > 0).sum(axis=0)   # (Ny,Nx)
    n_total         = int(ev.sum())

    return n_total, invasions_sp_px, richness_px
