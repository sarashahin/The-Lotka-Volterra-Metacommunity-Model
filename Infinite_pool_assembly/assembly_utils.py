############################################
# assembly_utils.py
############################################
from accelerator import np
import numpy as _real_numpy
from config import CONNECTANCE, INTERACTION_STRENGTH

def draw_interactions(S, rng=None):
    """
    Determines WHICH species interact (Topology).
    NOTE: This is the legacy topological function. 
    The new TraitManager logic handles topology implicitly via sparsity.
    """
    if rng is None: rng = _real_numpy.random
    mask_row = rng.random(S) < CONNECTANCE
    col_inds = _real_numpy.where(mask_row)[0]
    mask_col = rng.random(S) < CONNECTANCE
    row_inds = _real_numpy.where(mask_col)[0]
    return row_inds, col_inds

def expand_RC(r, C, new_r_vals, row_inds, col_inds, row_vals, col_vals):
    """
    Expands the r structure and C matrix.
    
    Args:
        r: Current growth rates. Can be (S,) or (S, P).
        C: Current interactions (S, S).
        new_r_vals: New species growth. (k,) or (k, P).
        row_inds: Indices of existing species that affect the new species (Incoming).
        col_inds: Indices of existing species affected by the new species (Outgoing).
        row_vals: Strength of effects ON new species.
        col_vals: Strength of effects BY new species.
    """
    k = len(new_r_vals)
    S = len(r)
    S_new = S + k
    
    # 1. Expand r (Handle both scalar vector and spatial matrix)
    if r.ndim == 2:
        # Spatial Field Case: (S, P) -> (S+k, P)
        n_patches = r.shape[1]
        r_out = np.zeros((S_new, n_patches), dtype=r.dtype)
        r_out[:S, :] = r
        r_out[S:, :] = new_r_vals
    else:
        # Legacy Scalar Case: (S,) -> (S+k,)
        r_out = np.zeros(S_new, dtype=r.dtype)
        r_out[:S] = r
        r_out[S:] = new_r_vals
    
    # 2. Expand C
    C_out = np.zeros((S_new, S_new), dtype=C.dtype)
    C_out[:S, :S] = C
    
    for i in range(k):
        C_out[S+i, S+i] = 1.0
        
    # 3. Fill Off-Diagonals
    if k == 1:
        # Row S: Interactions COMING IN to the New Species
        # C[New, Existing] = row_vals
        # We use row_inds (indices of Existing species acting on New)
        C_out[S, row_inds] = row_vals
        
        # Column S: Interactions GOING OUT from the New Species
        # C[Existing, New] = col_vals
        # We use col_inds (indices of Existing species affected by New)
        C_out[col_inds, S] = col_vals
    else:
        # Bulk addition logic (if needed in future)
        pass

    return r_out, C_out

def prune_extinct(alive_mask, r, C, *states):
    """
    Removes extinct species from r, C, and state arrays.
    """
    keep_inds = np.where(alive_mask)[0]
    
    # Handle r pruning (works for both 1D and 2D due to basic slicing)
    r_new = r[keep_inds]
    C_new = C[np.ix_(keep_inds, keep_inds)]
    
    new_states = []
    for s in states:
        if isinstance(s, tuple): # PSD2 state tuple
            B, W, PC = s
            new_states.append((B[keep_inds], W[keep_inds], PC[keep_inds]))
        elif s.ndim >= 1:
            new_states.append(s[keep_inds])
        else:
            new_states.append(s)
            
    return r_new, C_new, *new_states, keep_inds
