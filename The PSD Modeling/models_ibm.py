############################################
# models_ibm.py
############################################
"""
Individual-Based Model (IBM) approach for population dynamics.
using binomial & Poisson draws each step.
"""
import numpy as np
import logging
from config import (
    BODY_MASS,
    INV,
    MORTALITY_RATE,
    STEP_SIZE,
    TMAX,
    N_RECORDS,
    RECORDING_STEP_SIZE
)

logger = logging.getLogger(__name__)

class IBMModel:
    """
    IBM Model:
    - N: integer counts for each species
    - Convert to biomass by N * BODY_MASS
    - Growth rates: r - C@B
    """
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123):
        """
        :param r: 1D array of intrinsic growth rates (length S).
        :param C: 2D competition matrix (SxS).
        :param nsteps: number of steps in simulation (default TMAX).
        :param record_step: record every record_step steps.
        :param seed: random seed for reproducibility.
        """
        self.r = r
        self.C = C
        self.S = len(r)  # number of species
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        np.random.seed(seed)

        # Initialize counts (N)
        # In the R code, we had an initial 'BRelaxed' for each species.
        # Suppose each species starts with N[i], we do a small biomass:
        #we want them all to start at BODY_MASS/10 biomass => N = biomass / BODY_MASS
        init_biomass = BODY_MASS/10
        self.N = np.full(self.S, int(init_biomass / BODY_MASS), dtype=int)

        # Storage for trajectory
        self.nrecords = self.nsteps // self.record_step
        self.trajectory = np.full((self.nrecords, self.S), np.nan, dtype=float)

    def run(self):
        """
        Run the IBM simulation. 
        """
        logger.info("Starting IBM simulation...")
        record_idx = 0
        for s in range(self.nsteps):
            B = self.N * BODY_MASS
            # localGrowthRate = (r - C @ B)
            local_growth_rate = self.r - self.C.dot(B)

            # handle fastDying: localGrowthRate < - MORTALITY_RATE
            fast_dying = local_growth_rate < (-MORTALITY_RATE)
            full_mortality = np.full(self.S, MORTALITY_RATE)
            full_mortality[fast_dying] = -local_growth_rate[fast_dying]
            local_growth_rate[fast_dying] = -MORTALITY_RATE

            # Step
            # death
            survival_prob = np.exp(-full_mortality * STEP_SIZE)
            new_N = np.random.binomial(self.N, survival_prob)

            # birth
            birth_lambda = (np.exp((local_growth_rate + MORTALITY_RATE) * STEP_SIZE) - 1) * new_N
            birth_values = np.random.poisson(birth_lambda)

            # invasion
            invasion_values = np.random.poisson(INV * (STEP_SIZE / BODY_MASS) * np.ones(self.S))

            self.N = new_N + birth_values + invasion_values

            # Recording
            if (s+1) % self.record_step == 0:
                self.trajectory[record_idx, :] = self.N * BODY_MASS
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"IBM Progress: {record_idx}/{self.nrecords} records recorded.")

        logger.info("IBM simulation completed.")
        return self.trajectory