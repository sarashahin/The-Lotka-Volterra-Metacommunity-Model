############################################
# config.py
############################################
"""
Holds global constants and parameters used across the simulation.
"""
import math
from accelerator import np 

# -----------------------------
# Simulation Constants
# -----------------------------
BODY_MASS = 1e-4
INV = 1e-10
MORTALITY_RATE = 0.01  # 0.2

#: Initial number of species to aim for before the main loop
TARGET_RICHNESS = 300

#: Number of time steps for "relaxation"
T_RELAX = 300

# Probability of non-zero interaction
CONNECTANCE = 0.4

# Strength of interaction when non-zero
INTERACTION_STRENGTH = 0.4

#: Max simulation time
TMAX = 200000 

#: Step size per iteration
STEP_SIZE = 1

#: Recording step size
RECORDING_STEP_SIZE = 1000  
N_RECORDS = TMAX // RECORDING_STEP_SIZE

#: Random seed
RANDOM_SEED = 125

# For ODE/PSD2 solvers
RTOL = 1e-5
ATOL = 1e-4
MAX_STEPS = 100000

#: Threshold for "presence" in the system
THRESHOLD = 10 * BODY_MASS

# -----------------------------
# Spatial (Multi-patch) Parameters
# -----------------------------
NUM_PATCHES_X = 20
NUM_PATCHES_Y = 20

# Dispersal rate (diffusion coefficient)
DISPERSAL_RATE = BODY_MASS * 1e-3 
LONG_DISTANCE_PROB = 0 

# Ecological upper bound
ECOLOGICAL_MAX_B = 2 
LOG_B_CAP        = math.log(ECOLOGICAL_MAX_B)

# -----------------------------
# Dispersal Kernel (Long Distance)
# -----------------------------
# Function f(r) -> density. 
# If None, the system uses fast 3x3 Nearest-Neighbor convolution (Direct).
# If defined, the system switches to FFT-based convolution (Spectral).
# r is the distance in lattice units (0 to N/2).
#
# Example (Gaussian): 
# DISPERSAL_KERNEL = lambda r: np.exp(-r**2 / (2 * 2.0**2))
DISPERAL_A = 1
DISPERAL_B = 1.5  # 1.25
DISPERSAL_KERNEL = lambda r: (1 + (r/DISPERAL_A)**2)**(-DISPERAL_B)

