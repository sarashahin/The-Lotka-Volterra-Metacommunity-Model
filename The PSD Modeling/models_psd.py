############################################
# models_psd.py
############################################
"""
PSD approach for population dynamics.
 with discrete stepping and some logic for 'waiting' states.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PSDModel:
    """
    PSD Model with a discrete-time approach combining:
    - A 'waiting' state for small biomass species (B < BODY_MASS),
    - Poisson processes for invasion,
    - Deterministic growth for established species, etc.
    """
    def __init__(self, r, C, nsteps=None, record_step=None, seed=123):
        """
        :param r: 1D array of intrinsic growth rates.
        :param C: 2D competition matrix.
        """
        self.r = r
        self.C = C
        self.S = len(r)
        self.nsteps = nsteps if nsteps is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE

        np.random.seed(seed)
        
        # initial logB
        init_biomass = BODY_MASS/10
        self.logB = np.full(self.S, np.log(init_biomass))

        # waiting flags
        self.waiting = np.ones(self.S, dtype=bool)  # start them all as 'waiting' or not
        # In the R code, waiting starts if localGrowthRate>0 & B < bodyMass. 
        # We'll do the same initialization logic after we define them if needed.

        # PoissonClock
        # R code does PoissonClock = log(runif(S))
        self.poisson_clock = np.log(np.random.rand(self.S))

        # For data recording
        self.nrecords = self.nsteps // self.record_step
        self.trajectory = np.full((self.nrecords, self.S), np.nan, dtype=float)
        self.wait_trajectory = np.full((self.nrecords, self.S), np.nan, dtype=bool)

    def run(self):
        logger.info("Starting PSD simulation...")
        record_idx = 0

        for s in range(self.nsteps):
            B = np.exp(self.logB) * (~self.waiting)
            local_growth_rate = self.r - self.C.dot(B)

            # Re-classify waiting
            was_waiting = self.waiting.copy()
            # waiting if localGrowthRate>0 & B < BODY_MASS
            # But B might be zero if waiting was True
            # So effectively 'waiting' means we haven't established yet.
            # We'll do it as in the R code:
            self.waiting = (local_growth_rate > 0) & (B < BODY_MASS)
            new_waiting = np.where(self.waiting & (~was_waiting))[0]

            # handle new waiting species (PoissonClock logic)
            for nw in new_waiting:
                # establishmentProb = 1/(1 + mortalityRate/localGrowthRate[nw])
                est_prob = 1.0 / (1.0 + MORTALITY_RATE / local_growth_rate[nw]) if local_growth_rate[nw] != 0 else 0
                # increment Poisson clock
                self.poisson_clock[nw] += B[nw] / BODY_MASS * est_prob
                if self.poisson_clock[nw] > 0:
                    # repeated re-sampling until clock goes negative
                    n_est = 1
                    while True:
                        self.poisson_clock[nw] += np.log(np.random.rand())
                        if self.poisson_clock[nw] <= 0:
                            break
                        n_est += 1
                    # not waiting anymore
                    self.waiting[nw] = False
                    # logB[nw] = log(BODY_MASS * n_est / establishmentProb)
                    self.logB[nw] = np.log(BODY_MASS * n_est / est_prob)

            # handle species that stopped waiting this iteration
            stopped_waiting = np.where((~self.waiting) & was_waiting)[0]
            for sw in stopped_waiting:
                self.logB[sw] = np.log(INV * STEP_SIZE / 2)

            # Compute dB for established species
            dB = local_growth_rate + np.exp(np.log(INV) - self.logB)
            # handle waiting species: they won't grow or do the same in the step
            # (like dB=0 if waiting)
            dB[self.waiting] = 0

            # handle invasion events for waiting species
            waiting_list = np.where(self.waiting)[0]
            if len(waiting_list) > 0:
                est_prob_list = local_growth_rate[waiting_list] / (local_growth_rate[waiting_list] + MORTALITY_RATE)
                # invasionAndEstProb = i * (STEP_SIZE/BODY_MASS)*est_prob
                inv_est_prob = INV * (STEP_SIZE / BODY_MASS) * est_prob_list
                self.poisson_clock[waiting_list] += inv_est_prob
                # check which waiting species get triggered
                invades = np.where(self.poisson_clock[waiting_list] > 0)[0]
                if len(invades) > 0:
                    for i_idx in invades:
                        species_idx = waiting_list[i_idx]
                        # repeated re-sampling
                        n_est = 1
                        while True:
                            self.poisson_clock[species_idx] += np.log(np.random.rand())
                            if self.poisson_clock[species_idx] <= 0:
                                break
                            n_est += 1
                        # logB[i] = min(c(0, log(BODY_MASS * n_est / est_prob[i])))
                        # from R: we do "est_prob" ~ est_prob_list[i_idx]
                        self.logB[species_idx] = np.log(BODY_MASS * n_est / est_prob_list[i_idx])
                        if self.logB[species_idx] > 0:
                            self.logB[species_idx] = 0
                        # established half way => dB[i]/2
                        dB[species_idx] *= 0.5
                        self.waiting[species_idx] = False

            # update logB
            # update logB and enforce that established species (logB >= 0) are capped at 0
            self.logB += dB * STEP_SIZE
            self.logB = np.minimum(self.logB, 0)


            # Record
            if (s+1) % self.record_step == 0:
                self.trajectory[record_idx, :] = self.logB.copy()
                self.wait_trajectory[record_idx, :] = self.waiting.copy()
                record_idx += 1
                if record_idx % 10 == 0:
                    logger.info(f"PSD Progress: {record_idx}/{self.nrecords} records recorded.")

        logger.info("PSD simulation completed.")
        return self.trajectory, self.wait_trajectory
