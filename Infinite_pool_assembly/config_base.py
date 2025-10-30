############################################
# config_base.py
############################################
"""
Base configuration class containing common parameters used across all models.
Individual model configurations will inherit from this base class.
"""

import numpy as np

class BaseConfig:
    def __init__(self):
        # -----------------------------
        # Common Parameters
        # -----------------------------
        # Biomass threshold
        self.BODY_MASS = 1e-4
        
        # Random seed for reproducibility
        self.RANDOM_SEED = 123
        
        #: Threshold for "presence" in the system (biomass)
        self.THRESHOLD = 10 * self.BODY_MASS
        
        # -----------------------------
        # Spatial Configuration
        # -----------------------------
        # Number of patches
        self.NUM_PATCHES_X = 30
        self.NUM_PATCHES_Y = 30
        
        # -----------------------------
        # Community Parameters
        # -----------------------------
        # Competition parameters
        self.CONNECTANCE = 0.4
        self.INTERACTION_STRENGTH = 0.4
        
        # -----------------------------
        # Simulation Constants
        # -----------------------------
        #: Max simulation time
        self.TMAX = 10000
        
        #: Step size per iteration
        self.STEP_SIZE = 1
        
        #: Recording step size
        self.RECORDING_STEP_SIZE = 1000
        
        #: Number of records to keep
        self.N_RECORDS = self.TMAX // self.RECORDING_STEP_SIZE
        
        # -----------------------------
        # Dispersal Parameters
        # -----------------------------
        # Dispersal rate
        # self.DISPERSAL_RATE = 0.5
        
        # Wind parameters
        self.WIND_DIRECTION = np.pi/4  # 45 degrees in radians
        self.WIND_STRENGTH = 0.7
        self.WIND_X = self.WIND_STRENGTH * np.cos(self.WIND_DIRECTION)
        self.WIND_Y = self.WIND_STRENGTH * np.sin(self.WIND_DIRECTION)
        
        # Habitat quality parameters
        self.HABITAT_QUALITY = {
            'base': 1.0,
            'variation': 0.5,
            'scale': 3.0
        }
        
        # -----------------------------
        # Seed Size Parameters
        # -----------------------------
        # Seed size parameters for each species
        self.SEED_SIZES = [0.1, 0.5, 2.0]
        
        # Seed size effects on dispersal
        self.SEED_SIZE_EFFECTS = {
            'dispersal_distance': {
                'base': 1.2,
                'effect': -0.8
            },
            'establishment': {
                'base': 0.4,
                'effect': 0.6
            }
        }
        
        # -----------------------------
        # Dispersal Kernel Parameters
        # -----------------------------
        # Parameters for different kernels
        self.DISPERSAL_KERNEL_PARAMS = {
            'exponential': {
                'lambda': 0.8,
            },
            'gaussian': {
                'sigma': 1.5,
            },
            'levy': {
                'alpha': 1.2,
                'cutoff': 15.0
            }
        }
        
        # Species-specific dispersal parameters
        self.SPECIES_DISPERSAL_PARAMS = {
            'kernel_type': ['gaussian', 'exponential', 'levy'],
            'kernel_params': [
                {'sigma': 1.0},
                {'lambda': 0.5},
                {'alpha': 1.5, 'cutoff': 10.0}
            ]
        }
        
        
class IBMConfig(BaseConfig):
    """Configuration specific to the IBM model."""
    def __init__(self):
        super().__init__()
        # Override IBM-specific parameters
        self.TARGET_RICHNESS = 300
        self.T_RELAX = 300
        self.MAX_STEPS = 100000
        self.TMAX = 100000
        self.RECORDING_STEP_SIZE = 1000
        self.MORTALITY_RATE = 0.1
        self.INTERACTION_STRENGTH = 0.4
        self.DISPERSAL_RATE = 0.2
        self.WIND_STRENGTH = 0.7
        self.HABITAT_QUALITY['variation'] = 0.5
        self.HABITAT_QUALITY['scale'] = 3.0

class PSD2Config(BaseConfig):
    """Configuration specific to the PSD2 model."""
    def __init__(self):
        super().__init__()
        # PSD2-specific parameters
        self.INTERACTION_STRENGTH = 0.4  # competition strength
        self.MORTALITY_RATE = 0.1  # mortality rate
        self.DISPERSAL_RATE = 0.2   # dispersal rate for more spatial dynamics
        self.WIND_STRENGTH = 0.7    # wind effect
        self.HABITAT_QUALITY['variation'] = 0.5  #  variation
        self.HABITAT_QUALITY['scale'] = 3.0      #  spatial scale
        self.RTOL = 1e-8  #  precision
        self.ATOL = 1e-8  #  precision
        self.TMAX = 100000 #  simulation time
        self.RECORDING_STEP_SIZE = 1000  #  recording step size
        self.N_RECORDS = self.TMAX // self.RECORDING_STEP_SIZE #  number of records
        
        # Adjust dispersal kernel parameters for more realistic spatial patterns
        self.DISPERSAL_KERNEL_PARAMS['exponential']['lambda'] = 0.8
        self.DISPERSAL_KERNEL_PARAMS['gaussian']['sigma'] = 1.5
        self.DISPERSAL_KERNEL_PARAMS['levy']['alpha'] = 1.2
        self.DISPERSAL_KERNEL_PARAMS['levy']['cutoff'] = 15.0
        
        # Add new parameters for spatial analysis
        self.SPATIAL_ANALYSIS = {
            'autocorrelation_lags': 5,  # Number of lags for spatial autocorrelation
            'turnover_window': 100,     # Window size for calculating species turnover
            'patch_occupancy_threshold': 0.1  # Threshold for considering a patch occupied
        }

class ODEConfig(BaseConfig):
    """Configuration specific to the ODE model."""
    def __init__(self):
        super().__init__()
        # ODE-specific parameters
        self.RTOL = 1e-6
        self.ATOL = 1e-6
        self.TMAX = 10000
        self.RECORDING_STEP_SIZE = 100
        self.N_RECORDS = self.TMAX // self.RECORDING_STEP_SIZE
        self.MORTALITY_RATE = 0.1
        self.INTERACTION_STRENGTH = 0.4
        self.DISPERSAL_RATE = 0.2