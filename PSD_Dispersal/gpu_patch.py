# gpu_patch.py ----------------------------------------------------------
"""
Import this *before* anything else to replace the global `numpy` module
with CuPy when USE_GPU=1 is set in the environment.
Nothing happens on machines without a suitable GPU.
"""
import os, importlib, sys

if os.getenv("USE_GPU", "0") == "1":            # opt‑in switch
    np_gpu = importlib.import_module("cupy")    # CuPy is the drop‑in
    sys.modules["numpy"] = np_gpu               # hijack the name
    print("[gpu_patch] ✔ NumPy -> CuPy (GPU) enabled")
else:
    print("[gpu_patch] ⏩ running on CPU – set USE_GPU=1 to enable CUDA")
