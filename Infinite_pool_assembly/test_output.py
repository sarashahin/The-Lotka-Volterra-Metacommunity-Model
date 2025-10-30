############################################
# test_output.py
############################################

import numpy as np
import os
############################################


# Determine the base directory 
# BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# DATA_FILE = os.path.join(BASE_DIR, "results", "data", "pool22510000_batcha_ls2p5_vr0p001_thr3p0_env456_grid20x20_dr5em08_ld0p0_training.npz")
# data      = np.load(DATA_FILE, allow_pickle=True)



# def print_limited(key, arr, num=1):
#     print(f"Key: {key}")
#     print("Shape:", arr.shape)
#     if arr.ndim == 0:
#         # For scalars, just print the value.
#         print(arr)
#     elif arr.ndim == 1:
#         print(arr[:num])
#     elif arr.ndim == 2:
#         print(arr[:num, :])
#     else:
#         # For arrays with ndim >= 3, print the first 'num' elements along the first axis.
#         print(arr[:num])
#     print("\n" + "-"*50 + "\n")

# print("Available keys:", data.files)
# print("\n" + "="*50 + "\n")

# for key in data.files:
#     print_limited(key, data[key], num=1)




# --------------------------pick a good Stage-1 checkpoint---------------------
import torch, glob
for p in sorted(glob.glob("runs/stage1_sim/checkpoints/*.pt")):
    ck = torch.load(p, map_location="cpu")
    print(p, ck.get("extra",{}).get("val_metrics",{}))





