#!/usr/bin/env python3
# analyze_invasion_richness.py

"""
Compute chunk-based means ± standard errors for richness and invasion rates
for IBM, PSD2, and ODE models. Then perform t-tests to compare IBM vs PSD2 (and ODE).
"""

import numpy as np
import os
from scipy.stats import ttest_ind
import sys

# -------------------------------------------------------------------
# 1. PARAMETERS (edit as needed)
# -------------------------------------------------------------------

# Path to the saved .npz file from main.py
NPZ_PATH = "/home/sara/Downloads/The-Lotka-Volterra-Metacommunity-Model-main/The PSD Modeling/body-mass 1e-4 --inv 1e-10 --S 500 --seeds456/model_outputs.npz"  

# Invasion threshold (biomass cutoff for an “invasion” event)
B_th_invasion = 1e-3  

# Burn-in fraction: fraction of the early records to discard
burn_in_frac = 0.20  

# List of chunk counts to try (e.g. 5 and 10)
n_chunks_list = [5, 10]  

# We need RECORDING_STEP_SIZE and THRESHOLD (presence cutoff) from config.py

sys.path.append(os.getcwd())  # so that config.py is importable
from config import RECORDING_STEP_SIZE, THRESHOLD

record_interval = RECORDING_STEP_SIZE  
presence_thresh = THRESHOLD  

# -------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------

def compute_richness(trajectory, presence_thresh):
    """
    Given a trajectory array of shape (n_records, n_species),
    return a 1D array of richness at each record: count of species
    with biomass > presence_thresh.
    """
    # trajectory: shape = (n_records, S)
    return np.sum(trajectory > presence_thresh, axis=1)


def compute_invasions(trajectory, invasion_thresh):
    """
    Given a trajectory array of shape (n_records, n_species),
    return a 1D array inv_counts of length (n_records - 1),
    where inv_counts[t] = # species whose biomass crossed upward
    through invasion_thresh between record t and t+1.

    Assumes that at each record we know the biomass of all species.
    If a species was below invasion_thresh at record t and above
    at record t+1, that's one invasion.
    """
    above = (trajectory > invasion_thresh).astype(int)  # shape = (n_records, S), 1 if biomass > invasion_thresh
    # Upward crossings: from 0 → 1 between rows
    upward = (above[1:, :] == 1) & (above[:-1, :] == 0)  # boolean matrix of shape (n_records-1, S)
    inv_counts = np.sum(upward, axis=1)  # sum across species, shape = (n_records-1,)
    return inv_counts  # shape = (n_records-1,)


def chunked_statistics(ts_array, burn_in_frac, n_chunks, rec_step, mode):
    """
    Split a 1D time-series array (ts_array) into n_chunks, after discarding burn-in.
    Two modes:
      - "richness": each ts_array[t] is the richness at record t.
          chunk_value = mean richness in chunk (average of ts_array in chunk).
      - "invasion": each ts_array[t] is # of invasions between t and t+1.
          chunk_value = (sum of inv_counts in chunk) / (duration of chunk in time).
    Returns:
      mean_over_chunks, se_over_chunks, chunk_values (length = n_chunks).
    """
    N = len(ts_array)
    burn_cut = int(np.floor(burn_in_frac * N))
    ts_post = ts_array[burn_cut:]  # discard burn-in

    M = len(ts_post)
    if M < n_chunks:
        raise ValueError(f"After burn-in, only {M} points remain but n_chunks={n_chunks}.")

    # Compute indices for splitting into n_chunks as evenly as possible
    # E.g. if M=97 and n_chunks=5, chunk_sizes = [19,19,19,20,20].
    base_size = M // n_chunks
    remainder = M % n_chunks

    chunk_boundaries = []
    start = 0
    for i in range(n_chunks):
        size_i = base_size + (1 if i < remainder else 0)
        end_i = start + size_i
        chunk_boundaries.append((start, end_i))
        start = end_i

    chunk_values = np.zeros(n_chunks, dtype=float)

    for idx, (start, end) in enumerate(chunk_boundaries):
        segment = ts_post[start:end]
        if mode == "richness":
            # Mean richness = average value of ts_array in this chunk
            chunk_values[idx] = np.mean(segment)
        elif mode == "invasion":
            # Invasion‐rate = total invasions in chunk / (chunk_time_span)
            # Each entry of ts_post is “# invasions between record t and t+1.”
            # The number of records in this chunk is (end - start).  Each corresponds to one dt=rec_step.
            total_invasions = np.sum(segment)
            time_span = (end - start) * rec_step
            chunk_values[idx] = total_invasions / float(time_span)
        else:
            raise ValueError("mode must be 'richness' or 'invasion'")

    mean_val = np.mean(chunk_values)
    # Use ddof=1 for sample standard deviation
    se_val = np.std(chunk_values, ddof=1) / np.sqrt(n_chunks)

    return mean_val, se_val, chunk_values


# -------------------------------------------------------------------
# 3. MAIN ANALYSIS
# -------------------------------------------------------------------

def main():
    # 3.1 Load the saved trajectories
    data = np.load(NPZ_PATH)
    # Note: “PSD” in the .npz is PSD from models_psd (log→exp, with masking).
    #       “PSD2” is the detailed PSD2 trajectory. We will compare PSD2 (more accurate) with IBM.
    #       To compare the “plain” PSD vs ODE, replace "PSD2" with "PSD" below.
    traj_ibm = data["IBM"]    # shape = (n_records, S)
    traj_psd2 = data["PSD2"]  # shape = (n_records, S)
    traj_ode = data["ODE"]    # shape = (n_records+1, S)  (check indexing)

    # 3.2 Compute richness time‐series for each model
    rich_ibm = compute_richness(traj_ibm, presence_thresh)       # length = n_rec_ibm
    rich_psd2 = compute_richness(traj_psd2, presence_thresh)     # length = n_rec_psd2
    rich_ode = compute_richness(traj_ode, presence_thresh)       # length = n_rec_ode

    # 3.3 Compute invasion time‐series for each model
    inv_ibm = compute_invasions(traj_ibm, B_th_invasion)         # length = n_rec_ibm - 1
    inv_psd2 = compute_invasions(traj_psd2, B_th_invasion)       # length = n_rec_psd2 - 1
    inv_ode = compute_invasions(traj_ode, B_th_invasion)         # length = n_rec_ode - 1

    # 3.4 For each choice of n_chunks, compute chunked stats
    results = {}
    for n_chunks in n_chunks_list:
        results[n_chunks] = {}

        # 3.4.1 IBM richness
        mean_r_ibm, se_r_ibm, chunks_r_ibm = chunked_statistics(
            ts_array=rich_ibm,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="richness"
        )
        # 3.4.2 PSD2 richness
        mean_r_psd2, se_r_psd2, chunks_r_psd2 = chunked_statistics(
            ts_array=rich_psd2,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="richness"
        )
        # 3.4.3 ODE richness
        mean_r_ode, se_r_ode, chunks_r_ode = chunked_statistics(
            ts_array=rich_ode,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="richness"
        )

        # 3.4.4 IBM invasion rate
        mean_i_ibm, se_i_ibm, chunks_i_ibm = chunked_statistics(
            ts_array=inv_ibm,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="invasion"
        )
        # 3.4.5 PSD2 invasion rate
        mean_i_psd2, se_i_psd2, chunks_i_psd2 = chunked_statistics(
            ts_array=inv_psd2,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="invasion"
        )
        # 3.4.6 ODE invasion rate
        mean_i_ode, se_i_ode, chunks_i_ode = chunked_statistics(
            ts_array=inv_ode,
            burn_in_frac=burn_in_frac,
            n_chunks=n_chunks,
            rec_step=record_interval,
            mode="invasion"
        )

        results[n_chunks]["IBM"] = {
            "rich_mean": mean_r_ibm, "rich_se": se_r_ibm, "rich_chunks": chunks_r_ibm,
            "inv_mean" : mean_i_ibm, "inv_se" : se_i_ibm, "inv_chunks" : chunks_i_ibm,
        }
        results[n_chunks]["PSD2"] = {
            "rich_mean": mean_r_psd2, "rich_se": se_r_psd2, "rich_chunks": chunks_r_psd2,
            "inv_mean" : mean_i_psd2, "inv_se" : se_i_psd2, "inv_chunks" : chunks_i_psd2,
        }
        results[n_chunks]["ODE"] = {
            "rich_mean": mean_r_ode, "rich_se": se_r_ode, "rich_chunks": chunks_r_ode,
            "inv_mean" : mean_i_ode, "inv_se" : se_i_ode, "inv_chunks" : chunks_i_ode,
        }

    # 3.5 Perform t-tests between IBM and PSD2 (and IBM vs ODE, PSD2 vs ODE) for each n_chunks
    ttest_results = {}
    for n_chunks in n_chunks_list:
        ttest_results[n_chunks] = {}

        # Extract chunk‐wise arrays
        chunks_r_ibm = results[n_chunks]["IBM"]["rich_chunks"]
        chunks_r_psd2 = results[n_chunks]["PSD2"]["rich_chunks"]
        chunks_r_ode = results[n_chunks]["ODE"]["rich_chunks"]

        chunks_i_ibm = results[n_chunks]["IBM"]["inv_chunks"]
        chunks_i_psd2 = results[n_chunks]["PSD2"]["inv_chunks"]
        chunks_i_ode = results[n_chunks]["ODE"]["inv_chunks"]

        # Richness: IBM vs PSD2
        t_r_ibm_psd2, p_r_ibm_psd2 = ttest_ind(chunks_r_ibm, chunks_r_psd2, equal_var=False)
        # Richness: IBM vs ODE
        t_r_ibm_ode, p_r_ibm_ode = ttest_ind(chunks_r_ibm, chunks_r_ode, equal_var=False)
        # Richness: PSD2 vs ODE
        t_r_psd2_ode, p_r_psd2_ode = ttest_ind(chunks_r_psd2, chunks_r_ode, equal_var=False)

        # Invasion: IBM vs PSD2
        t_i_ibm_psd2, p_i_ibm_psd2 = ttest_ind(chunks_i_ibm, chunks_i_psd2, equal_var=False)
        # Invasion: IBM vs ODE
        t_i_ibm_ode, p_i_ibm_ode = ttest_ind(chunks_i_ibm, chunks_i_ode, equal_var=False)
        # Invasion: PSD2 vs ODE
        t_i_psd2_ode, p_i_psd2_ode = ttest_ind(chunks_i_psd2, chunks_i_ode, equal_var=False)

        ttest_results[n_chunks]["richness"] = {
            ("IBM", "PSD2"): (t_r_ibm_psd2, p_r_ibm_psd2),
            ("IBM", "ODE") : (t_r_ibm_ode, p_r_ibm_ode),
            ("PSD2", "ODE"): (t_r_psd2_ode, p_r_psd2_ode),
        }
        ttest_results[n_chunks]["invasion"] = {
            ("IBM", "PSD2"): (t_i_ibm_psd2, p_i_ibm_psd2),
            ("IBM", "ODE") : (t_i_ibm_ode, p_i_ibm_ode),
            ("PSD2", "ODE"): (t_i_psd2_ode, p_i_psd2_ode),
        }

    # 3.6 Print out a summary
    print("\n=================== SUMMARY: Means ± SE ===================\n")
    for n_chunks in n_chunks_list:
        print(f"--- n_chunks = {n_chunks} ---")
        for model in ["IBM", "PSD2", "ODE"]:
            r_mean = results[n_chunks][model]["rich_mean"]
            r_se   = results[n_chunks][model]["rich_se"]
            i_mean = results[n_chunks][model]["inv_mean"]
            i_se   = results[n_chunks][model]["inv_se"]
            print(f"{model:4s} | Richness = {r_mean:.2f} ± {r_se:.2f}  | Invasion‐rate = {i_mean:.2e} ± {i_se:.2e}")
        print()

    print("\n=================== T‐TEST RESULTS ===================\n")
    for n_chunks in n_chunks_list:
        print(f"--- n_chunks = {n_chunks} ---")
        print(" * Richness comparisons:")
        for (m1, m2), (tval, pval) in ttest_results[n_chunks]["richness"].items():
            print(f"   {m1} vs {m2}: t = {tval:.3f}, p = {pval:.3e}")
        print(" * Invasion‐rate comparisons:")
        for (m1, m2), (tval, pval) in ttest_results[n_chunks]["invasion"].items():
            print(f"   {m1} vs {m2}: t = {tval:.3f}, p = {pval:.3e}")
        print()

    print("Done.\n")


if __name__ == "__main__":
    main()
