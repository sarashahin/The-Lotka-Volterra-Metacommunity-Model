############################################
# config.py
############################################
"""
Holds global constants and parameters used across the simulation.
"""

import numpy as np

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
NUM_PATCHES_X = 3   # Number of patches horizontally
NUM_PATCHES_Y = 3   # Number of patches vertically

# Dispersal rate (diffusion coefficient): fraction of biomass exchanged per time step.
DISPERSAL_RATE = 0.001

# x = np.linspace(0.5, 1.5, NUM_PATCHES_X)
# y = np.linspace(0.5, 1.5, NUM_PATCHES_Y)
# X, Y = np.meshgrid(x, y)
# DISPERSAL_FIELD = 0.001 * (X + Y) / 2
DISPERSAL_FIELD = None

