############################################
# utils.py
############################################
"""
Common utility functions: moving average, blockwise average, mean+SE, logging, etc.
"""
import numpy as np
import logging
import sys

def setup_logging():
    """
    Sets up Python's logging module to print to console with level=INFO.
    Call once at the start of the program.
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def mav(x, n=100):
    """
    Moving average (window size = n). 
    If len(x) < n, adjust n to be len(x).
    Similar to the R function in the snippet.
    """
    x = np.asarray(x, dtype=float)
    length = len(x)
    if length == 0:
        return x  # empty
    if n > length:
        # Adjust
        n = length
    # np.convolve for simpler moving average approach
    # We do 'same' mode to keep the size consistent
    window = np.ones(int(n))/float(n)
    out = np.convolve(x, window, 'same')
    return out

def bav(x, n=50):
    """
    Blockwise average of x, block size n.
    """
    x = np.asarray(x, dtype=float)
    length = len(x)
    if length < n:
        return np.array([np.nan])
    num_blocks = length // n
    if num_blocks < 1:
        return np.array([np.nan])
    # Truncate to multiple of n
    valid_length = n * num_blocks
    x_truncated = x[:valid_length]
    M = x_truncated.reshape(num_blocks, n)
    # colMeans in R ~ mean along axis=1 in Python
    return np.mean(M, axis=1)

def mean_se(v):
    """
    Return (mean, standard error) ignoring NaNs.
    If all values are NaN, return (NaN, NaN).
    """
    arr = np.asarray(v, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    mean_val = np.mean(arr)
    if len(arr) > 1:
        se_val = np.std(arr, ddof=1) / np.sqrt(len(arr))
    else:
        se_val = np.nan
    return (mean_val, se_val)
