############################################
# config.py
############################################
"""
Holds global constants and parameters used across the simulation.
"""
import numpy as np
import gpu_patch

# -----------------------------
# Simulation Constants
# -----------------------------

# Biomass unit
BODY_MASS = 1e-4

# BODY_MASS = 1e-11

# Invasion rate
# INV = 1e-10
INV = 1e-8
# Mortality rate (for large negative growth offsets)
MORTALITY_RATE = 0.2

#: Initial number of species to aim for before the main loop
TARGET_RICHNESS = 300

#: Number of time steps for "relaxation" or "settling" in certain procedures
T_RELAX = 300

# Probability of non-zero interaction
CONNECTANCE = 0.4

# Strength of interaction when non-zero
INTERACTION_STRENGTH = 0.4

#: Max simulation time
TMAX = 20000  # (like 200000 in R)100000

#: Step size per iteration in the simpler discrete loops
STEP_SIZE = 1

#: Recording step size
RECORDING_STEP_SIZE = 100  # e.g., 1000

#: Number of records to keep
N_RECORDS = TMAX // RECORDING_STEP_SIZE

#: Random seed for reproducibility
RANDOM_SEED = 123

# For ODE/PSD2 solvers
RTOL = 1e-5
ATOL = 1e-4
MAX_STEPS = 100000

#: Threshold for "presence" in the system (biomass)
THRESHOLD = 10 * BODY_MASS

# -----------------------------
# Spatial (Multi-patch) Parameters
# -----------------------------

# Define a 2D grid of patches.
NUM_PATCHES_X = 50 # Number of patches horizontally
NUM_PATCHES_Y = 50   # Number of patches vertically

# Dispersal rate (diffusion coefficient): fraction of biomass exchanged per time step.

# DISPERSAL_RATE = BODY_MASS * 0.0025 # half rate is 0.0025,Axel is expected,only for nonlocal  oscillation 2.5e-7
DISPERSAL_RATE = BODY_MASS * 0.005 # full rate is 0.005,Axel is expected,only nonlocal  oscillation 5e-7
# LONG_DISTANCE_PROB = 0  # Probability of local-distance dispersal

LONG_DISTANCE_PROB = 1.0  # Probability of long-distance dispersal- only nonlocal  oscillation 






