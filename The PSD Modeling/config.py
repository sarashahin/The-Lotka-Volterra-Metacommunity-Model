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
INV = 1e-8

# Mortality rate (for large negative growth offsets)
MORTALITY_RATE = 0.2

# Probability of non-zero interaction
CONNECTANCE = 0.4

# Strength of interaction when non-zero
INTERACTION_STRENGTH = 0.4

#: Initial number of species to aim for before the main loop
TARGET_RICHNESS = 300

#: Number of time steps for "relaxation" or "settling" in certain procedures
T_RELAX = 300

#: Max simulation time
TMAX =  200000  # (like 200000 in R)

#: Step size per iteration in the simpler discrete loops
STEP_SIZE = 1

#: Recording step size
RECORDING_STEP_SIZE = 1000  # e.g., 1000

#: Number of records to keep
N_RECORDS = TMAX // RECORDING_STEP_SIZE

#: We can store random seed for reproducibility
RANDOM_SEED = 123

# For PSD2 or ODE solver
# Some ODE solver tolerances
RTOL = 1e-5
ATOL = 1e-4

#: Threshold for "presence" in the system (biomass)
THRESHOLD = 10 * BODY_MASS

# We also define maximum steps for Assimulo
MAX_STEPS = 100000

# -----------------------------
# Consumer Parameters
# -----------------------------
NUM_CONSUMERS = 1
TOTAL_SPECIES = TARGET_RICHNESS + NUM_CONSUMERS

ATTACK_RATE = 0.3             # Base attack rate for consumer on resources
SIGMA = 0.6                  # Sigma parameter for consumer dietary breadth
CONVERSION_EFFICIENCY = 0.1    # Conversion efficiency for consumer biomass gain
RESPIRATION_RATE = 0.1         # Respiration rate for consumers
