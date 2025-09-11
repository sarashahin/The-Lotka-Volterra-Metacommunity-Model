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
BODY_MASS = 1e-4 # << large body mass
# BODY_MASS = 1e-11 # << small body mass 

INV = 1e-10          # << tiny immigration 101

# Mortality rate (for large negative growth offsets)
MORTALITY_RATE = 0.2

# Probability of non-zero interaction
CONNECTANCE = 0.4

# Strength of interaction when non-zero
INTERACTION_STRENGTH = 0.4

#: Initial number of species to aim for before the main loop
TARGET_RICHNESS = 500

#: Number of time steps for "relaxation" or "settling" in certain procedures
T_RELAX = 300

#: Max simulation time
TMAX =  200000  # (like 200000 in R)

#: Step size per iteration in the simpler discrete loops
STEP_SIZE = 1

#: Recording step size
RECORDING_STEP_SIZE = 200  # e.g., 500

#: Number of records to keep
N_RECORDS = TMAX // RECORDING_STEP_SIZE

#: We can store random seed for reproducibility
RANDOM_SEED = 456 # << large body mass
# RANDOM_SEED = 123 # << small body mass


# For PSD2 or ODE solver
# Some ODE solver tolerances
RTOL = 1e-6 #1e-5
ATOL = 1e-7 #1e-4

#: Threshold for "presence" in the system (biomass)
THRESHOLD = 10 * BODY_MASS


# We also define maximum steps for Assimulo
MAX_STEPS = 100000


