############################################
# assembly_utils.py
############################################

import numpy as np
from config import CONNECTANCE, INTERACTION_STRENGTH

def draw_interactions(k: int, *, rng=np.random):
    """
    Draw a size-(k,) vector of inter-specific coefficients for ONE new species
    from the Jack-style distribution.
    Returns (new_row, new_col) where

      new_row[j]  = effect of resident j ON the new species
      new_col[j]  = effect of the new species ON resident j
    """
    mask = rng.random(k) < CONNECTANCE
    new_row = np.zeros(k)
    new_col = np.zeros(k)
    new_row[mask] = INTERACTION_STRENGTH * rng.random(mask.sum())
    mask = rng.random(k) < CONNECTANCE
    new_col[mask] = INTERACTION_STRENGTH * rng.random(mask.sum())
    return new_row, new_col

def expand_RC(r: np.ndarray, C: np.ndarray, r_new: float,
              row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append one species to (r,C) and keep C_ii = 1."""
    r_out = np.append(r, r_new)
    k = len(r)
    C_out = np.empty((k+1, k+1), float)
    C_out[:k, :k] = C
    C_out[k, :k]  = row          # new last row  (j → new)
    C_out[:k, k]  = col          # new last col  (new → j)
    C_out[k, k]   = 1.0
    return r_out, C_out

def prune_extinct(mask_alive: np.ndarray,
                  r: np.ndarray, C: np.ndarray, *states):
    """Remove species where mask_alive == False, returns pruned copies."""
    keep = np.where(mask_alive)[0]
    r2   = r[keep]
    C2   = C[np.ix_(keep, keep)]
    sts  = [s[keep] if s.shape[0]==len(mask_alive) else s for s in states]
    return (r2, C2, *sts, keep)
