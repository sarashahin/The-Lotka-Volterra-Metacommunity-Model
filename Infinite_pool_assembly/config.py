############################################
# config.py
############################################
"""
Holds global constants and parameters used across the simulation.
"""
import numpy as np
import gpu_patch
import math
import gpu_patch as gpu

# -----------------------------
# Simulation Constants
# -----------------------------

# Biomass unit
BODY_MASS = 1e-4

# Invasion rate
INV = 1e-10

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
TMAX = 200000  # (like 200000 in R)

#: Step size per iteration in the simpler discrete loops
STEP_SIZE = 1

#: Recording step size
RECORDING_STEP_SIZE = 1000  # e.g., 1000

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
NUM_PATCHES_X = 20 # Number of patches horizontally
NUM_PATCHES_Y = 20
  # Number of patches vertically

# Dispersal rate (diffusion coefficient): fraction of biomass exchanged per time step.
DISPERSAL_RATE = BODY_MASS * 0.0002 # 0.00002 for 25 patches, 0.002 for 5 patches
# DISPERSAL_RATE = 0.02
# LONG_DISTANCE_PROB = 1  # Probability of long-distance dispersal
# DISPERSAL_RATE = BODY_MASS * 0.005
LONG_DISTANCE_PROB = 0  # Probability of long-distance dispersal



# ecological upper bound: one adult body mass
ECOLOGICAL_MAX_B = 1 # one adult biomass
LOG_B_CAP        = math.log(ECOLOGICAL_MAX_B)


