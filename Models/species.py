# -*- coding: utf-8 -*-
"""Species.py

Updated and completed version with all missing methods implemented.
"""

from topography import Topography  # Importing Topography class
from lvmcm_rng import LVMCM_rng



import numpy as np
import logging
from typing import Optional, Tuple, List
from scipy.io import loadmat
from scipy.stats import binom


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Species:
    def __init__(self, topo: 'Topography', scVec=None, dMat=None, tMat=None, c1=0.1, c2=0.2, c3=-1.0, efMat=None, emRate=0.05, dispL=1.0, 
                 pProducer=0.5, prodComp=True, symComp=False, alpha=0.5, sigma=0.1, sigma_t=0.05, rho=0.01, comp_dist=0, omega=0.5, 
                 dispNorm=0, delta_g=1.25, bodymass=1e-4, mu=0.1):
        self.topo = topo
        no_nodes = self.topo.no_nodes

        # Initialize species-related matrices with zero species
        self.xMat = np.zeros((0, no_nodes))
        self.rMat = np.zeros((0, no_nodes))
        self.cMat = np.zeros((0, 0))
        self.dMat = dMat if dMat is not None else np.zeros((no_nodes, no_nodes))
        self.efMat = np.zeros((0, no_nodes))
        self.ouMat = np.zeros((0, no_nodes))
        self.sMat = np.zeros((0, no_nodes))
        # Ensure tMat has shape (0,2) if not provided
        # We need at least 2 columns for temperature optimum and scaling factor
        self.tMat = tMat if tMat is not None else np.zeros((0, 2))
        self.trajectories = np.zeros((0, no_nodes))
        self.fluctuations = np.zeros((0, no_nodes))
        logger.debug(f"Initialized sMat with shape: {self.sMat.shape}")
        self.emMat = np.zeros((0, no_nodes))

        # Parameters
        self.c1 = c1  # interspecific competition parameter 1
        self.c2 = c2  # interspecific competition parameter 2
        self.c3 = c3  # interspecific competition parameter 3
        # Initialize invasion counter
        self.invasion = 0
        self.rho = rho  # consumer mortality
        self.sigma = sigma  # standard deviation of attack rate distribution
        self.alpha = alpha  # base attack rate
        self.pProducer = pProducer  # probability of invading producer species
        self.emRate = emRate  # emigration rate
        self.emMat = np.array([])  # To be updated as species are added
        self.dispL = dispL  # dispersal length
        self.thresh = 1e-3  # detection/extinction threshold
        self.sigma_t = sigma_t  # temporal autocorrelation (OU process)
        self.omega = omega  # temperature niche width
        self.delta_g = delta_g  # range of environmental optima
        self.bodymass = bodymass  # body mass
        self.bodymass_inv = 1 / self.bodymass  # inverse body mass
        self.mu = mu  # mortality
        self.sppRichness = np.empty((0, 2))  # Each row: [S_p, S_c]
        # Initialize scVec and scVec_prime as empty arrays++
        # If scVec was given non-empty, it must start empty or raise an error
        if scVec is not None and scVec.size != 0:
            raise ValueError("scVec must be empty at initialization.")
        self.scVec = np.ones((0, no_nodes))  # start empty, rows added per species
        self.scVec_prime = np.zeros((0, no_nodes))


        # Initialize indices_S and indices_DP as empty 1D NumPy arrays
        self.indices_S = np.array([], dtype=int)  # For species indices (e.g., consumers)
        self.indices_DP = np.array([], dtype=int)  # For derived parameters or specific indices

        # Switches
        self.prodComp = prodComp  # producer competition
        self.comp_dist = comp_dist  # competition distribution type
        self.symComp = symComp  # symmetric competition
        self.dispNorm = dispNorm  # dispersal normalization method

        # Ensure dispersal matrix is generated during initialization
        self.gen_disp_mat()

        # Counters
        self.S_p = 0  # Producer species richness
        self.S_c = 0  # Consumer species richness
        self.I_p = 0  # Multispecies invasion counter (producers)
        self.I_c = 0  # Multispecies invasion counter (consumers)
        
        # If no producers, add default producer
        if self.S_p == 0:
            logger.info("Initializing with default invader as no producers exist.")
            self.invade(0)  # Add one default producer to ensure rMat is not empty
        
        # Initialize rMat with a default growth rate if there are no producers yet
        if self.S_p > 0:
            self.rMat = np.array([self.gen_r_vec() for _ in range(self.S_p)])
        else:
            logger.info("Initializing with default invader as no producers exist.")
            self.invade(0)  # Adding a default producer to ensure rMat is not empty

        # Ensure sMat and rMat dimensions align
        if self.rMat.shape != self.sMat.shape:
            logger.warning("sMat and rMat dimensions do not align. Resizing sMat.")
            self.sMat = np.tile(self.sMat, (self.rMat.shape[0], 1))

        # Validate dimensions of xMat and rMat
        if self.rMat.shape != self.xMat.shape:
            logger.warning("rMat and xMat dimensions do not align. Resizing xMat.")
            self.xMat = np.zeros_like(self.rMat)

        if self.tMat is None:
            self.tMat = np.zeros((0, self.topo.no_nodes))  # Initialize as empty matrix
        
        # Validate dimensions
        total_species = self.S_p + self.S_c
        if total_species > 0:
            assert self.xMat.shape == (total_species, no_nodes)
            assert self.rMat.shape == (total_species, no_nodes)
            assert self.cMat.shape == (total_species, total_species)
            assert self.sMat.shape == (total_species, no_nodes)
            assert self.scVec.shape == (total_species, no_nodes)
            assert self.scVec_prime.shape == (total_species, no_nodes)
            assert self.tMat.shape == (total_species, 2), "tMat must have 2 columns."


    def gen_r_vec(self, z_vec_ext=None):
        """
        Generate a spatially correlated random field for growth rates.

        Parameters:
        z_vec_ext (np.ndarray): Optional vector of random variables to control randomness.

        Returns:
        np.ndarray: Spatially autocorrelated and biologically meaningful growth rate vector.
        """
        logger.debug("Generating spatially correlated random growth vector.")
        
        desired_mean = 1.0  # Positive mean to ensure producers can grow
        desired_std = 0.5    # Standard deviation

        no_nodes = self.topo.no_nodes

        # Generate or use the provided random vector
        if z_vec_ext is None:
            z_vec = np.random.randn(no_nodes)  # Must be length no_nodes
            logger.debug("Generated new random vector for spatial correlation.")
        else:
            z_vec = z_vec_ext
            if z_vec.shape[0] != no_nodes:
                raise ValueError(f"z_vec_ext length {z_vec.shape[0]} does not match no_nodes={no_nodes}")
            logger.debug("Using external random vector for spatial correlation.")

        if not hasattr(self.topo, 'sigEVec') or not hasattr(self.topo, 'sigEVal'):
            raise ValueError("Topography eigen decomposition not initialized. Ensure sigEVec and sigEVal exist.")

        # Ensure sigEVal is a 1D array
        sigEVal = self.topo.sigEVal
        if sigEVal.ndim != 1:
            # If sigEVal is diagonal stored in another form, extract the diagonal
            # For example, if sigEVal is (N,N), use np.diag:
            if sigEVal.ndim == 2 and sigEVal.shape[0] == sigEVal.shape[1]:
                sigEVal = np.diag(sigEVal)
            else:
                raise ValueError("sigEVal must be a 1D array or a square diagonal matrix.")

        if sigEVal.shape[0] != no_nodes:
            raise ValueError(f"sigEVal length {sigEVal.shape[0]} does not match no_nodes={no_nodes}")

        # Perform element-wise multiplication (N,) * (N,) -> (N,)
        scaled_z = sigEVal * z_vec  # Result should be shape (no_nodes,)

        # sigEVec should be (no_nodes, no_nodes) and scaled_z is (no_nodes,)
        r_i = self.topo.sigEVec @ scaled_z  # Result: (no_nodes,)

        if r_i.ndim != 1 or r_i.shape[0] != no_nodes:
            raise ValueError(f"r_i must be a 1D vector of length {no_nodes}, got shape {r_i.shape}")

        current_mean = np.mean(r_i)
        current_std = np.std(r_i)
        if current_std == 0:
            logger.error("Standard deviation of growth rates is zero. Cannot normalize.")
            raise ValueError("Standard deviation of growth rates is zero.")

        r_i = (r_i - current_mean) / current_std
        r_i = r_i * desired_std + desired_mean
        logger.debug(f"Shifted and scaled growth vector: mean={np.mean(r_i):.4f}, std={np.std(r_i):.4f}")

        min_growth = 0.0
        max_growth = 3.0
        r_i = np.clip(r_i, min_growth, max_growth)
        logger.debug(f"Clipped growth vector to range [{min_growth}, {max_growth}]: {r_i}")

        # Now r_i is (no_nodes,) and can be reshaped to (1, no_nodes)
        return r_i


    def update_r_vec_temp(self):
        """
        Update rMat to reflect abiotic changes due to temperature shifts.
        """
        logger.debug("Updating growth rate vector for temperature changes.")
        self.topo.gen_temp_grad()

        # Ensure matrices are initialized
        if self.sMat.size == 0 or self.topo.envMat.size == 0 or self.tMat.size == 0:
            raise ValueError("One or more matrices (sMat, envMat, tMat) are not initialized or empty.")

        no_nodes = self.topo.no_nodes
        total_species = self.sMat.shape[0]

        if self.topo.envMat.shape[0] != no_nodes:
            raise ValueError(f"envMat rows ({self.topo.envMat.shape[0]}) do not match number of nodes ({no_nodes}).")

        if self.tMat.shape[0] != total_species:
            raise ValueError(f"tMat rows ({self.tMat.shape[0]}) do not match number of species ({total_species}).")

        envMat_SN = self.topo.envMat[:, np.newaxis]  # Ensure 2D structure
        tMat_SN = self.tMat[:, 0][:, np.newaxis]  # Extract temperature optima

        if envMat_SN.shape[0] != tMat_SN.shape[0]:
            raise ValueError(f"envMat_SN rows ({envMat_SN.shape[0]}) do not match tMat_SN rows ({tMat_SN.shape[0]}).")

        logger.debug(f"sMat shape: {self.sMat.shape}, envMat_SN shape: {envMat_SN.shape}, tMat_SN shape: {tMat_SN.shape}")

        try:
            rMat_cc = self.sMat.copy()

            for i in range(total_species):
                for j in range(no_nodes):
                    if self.omega > 0:
                        rMat_cc[i, j] -= self.omega * (envMat_SN[j, 0] - tMat_SN[i, 0]) ** 2
                    else:
                        rMat_cc[i, j] -= (envMat_SN[j, 0] - tMat_SN[i, 0]) ** 2 * self.tMat[i, 1]

            # Clip negative values to -1
            rMat_cc = np.clip(rMat_cc, -1, None)
            self.rMat = rMat_cc
            logger.debug("Updated rMat for temperature changes.")
        except Exception as e:
            logger.error(f"Error during rMat computation: {e}")
            raise


    def gen_r_vec_quad(self):
        """
        Generate growth rate vector based on quadratic environmental response.

        Returns:
        np.ndarray: Quadratic environmental response growth rate vector.
        """
        logger.debug("Generating growth rate vector based on quadratic environmental response.")
        g_ik = np.zeros(self.topo.envVar)
        for k in range(self.topo.envVar):
            g_i = np.random.rand() * self.topo.range_env[k]
            g_i += self.topo.min_env[k]
            g_i *= self.delta_g
            g_ik[k] = g_i

        r_i = np.ones(self.topo.no_nodes)
        for k in range(self.topo.envVar):
            emg = self.topo.envMat[k, :] - g_ik[k]
            r_i -= self.topo.skVec[k] * (emg ** 2)

        r_i = np.clip(r_i, -1, None)

        if self.tMat is None:
            self.tMat = g_ik[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, g_ik])

        logger.debug(f"Quadratic growth vector: {r_i}")
        return r_i


    def update_r_vec_temp(self):
        """
        Update `rMat` to reflect abiotic changes due to temperature shifts.
        """
        logger.debug("Updating growth rate vector for temperature changes.")
        self.topo.gen_temp_grad()

        # Validate matrix initializations
        if self.sMat is None or self.sMat.size == 0:
            raise ValueError("sMat is not initialized or empty.")
        if self.topo.envMat is None or self.topo.envMat.size == 0:
            raise ValueError("envMat is not initialized or empty.")
        if self.tMat is None or self.tMat.size == 0:
            raise ValueError("tMat is not initialized or empty.")

        # Determine dimensions
        S = self.rMat.shape[0]       # Number of species
        no_nodes = self.rMat.shape[1]  # Number of nodes

        # Create envMat_SN with shape (S, no_nodes)
        # envMat is (no_nodes,) so reshape to (1, no_nodes) and tile S times
        envMat_SN = np.tile(self.topo.envMat[np.newaxis, :], (S, 1))

        # tMat: (S, 2), use tMat[:,0] as temperature optimum, repeat across nodes
        tMat_SN = np.tile(self.tMat[:, 0][:, np.newaxis], (1, no_nodes))

        logger.debug(f"sMat shape: {self.sMat.shape}, envMat_SN shape: {envMat_SN.shape}, tMat_SN shape: {tMat_SN.shape}")

        # Compute growth rate changes
        try:
            if self.omega > 0:
                # (envMat_SN - tMat_SN) is (S,no_nodes), same as sMat
                rMat_cc = self.sMat - self.omega * (envMat_SN - tMat_SN) ** 2
            else:
                rMat_t = (envMat_SN - tMat_SN) ** 2
                # Use scaling factor from tMat[i,1] if available
                for i in range(rMat_t.shape[0]):
                    rMat_t[i] *= self.tMat[i, 1]
                rMat_cc = self.sMat - rMat_t

            # Clip negative values to -1
            rMat_cc[rMat_cc < 0] = -1
            self.rMat = rMat_cc
            logger.debug("Updated `rMat` for temperature changes.")
        except Exception as e:
            logger.error(f"Error during `rMat` computation: {e}")
            raise

        

    def ou_process(self):
        """
        Simulate abiotic turnover using an Ornstein-Uhlenbeck process.
        """
        logger.debug("Running Ornstein-Uhlenbeck process for abiotic turnover.")

        if self.ouMat is None or self.ouMat.shape != self.xMat.shape:
            self.ouMat = np.zeros_like(self.xMat)
        if self.efMat is None or self.efMat.shape != self.xMat.shape:
            self.efMat = np.zeros_like(self.xMat)

        zMat = np.random.randn(*self.efMat.shape)
        self.ouMat = (self.ouMat + self.sigma_t * zMat) / np.sqrt(1 + self.sigma_t ** 2)

        for i in range(self.rMat.shape[0]):
            self.efMat[i] = self.gen_r_vec()  # Use a valid function to generate growth rates

        logger.debug("Updated `ouMat` and `efMat` using Ornstein-Uhlenbeck process.")


    def gen_r_vec_erf(self):
        """
        Generate growth rates based on environmental response function (ERF).
        """
        logger.debug("Generating growth rates using Environmental Response Function (ERF).")
        tol = np.random.randn(self.topo.env_var)
        tol /= np.sqrt(np.sum(tol**2)) * np.sqrt(self.topo.env_var)

        r_i = 1 + np.dot(tol, self.topo.envMat) / np.sqrt(2 * self.topo.env_var)
        r_i[r_i < -1] = -1  # Clip values below -1

        if self.tMat is None:
            self.tMat = tol[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, tol])

        logger.debug(f"Generated `r_i`: {r_i}")
        return r_i



    def invade(self, trophLev):
        """
        Introduce an invader species into the system with consistent initialization 
        and interaction matrix updates.

        Parameters:
        trophLev (int): Trophic level of the invader (0 for producer, 1 for consumer).
        """
        logger.debug(f"Invading with trophic level {trophLev}.")
        inv_biomass = 1e-2  # Initial biomass for the invader species
        no_nodes = self.topo.no_nodes

        # # Ensure matrices are initialized
        # if self.xMat is None or self.xMat.size == 0:
        #     self.xMat = np.zeros((0, no_nodes))
        # if self.rMat is None or self.rMat.size == 0:
        #     self.rMat = np.zeros((0, no_nodes))
        # if self.cMat is None or self.cMat.size == 0:
        #     self.cMat = np.zeros((0, 0))
        # if self.sMat is None or self.sMat.size == 0:
        #     self.sMat = np.zeros((0, no_nodes))
        # if self.scVec is None or self.scVec.size == 0:
        #     self.scVec = np.ones((0, no_nodes))
        # if self.scVec_prime is None or self.scVec_prime.size == 0:
        #     self.scVec_prime = np.zeros((0, no_nodes))
        # if self.tMat is None or self.tMat.size == 0:
        #     # Initialize tMat as (0,2) to store temperature params for each species
        #     # column 0: temperature optimum, column 1: scaling factor
        #     self.tMat = np.zeros((0, 2))

        # Add a new species row to xMat and rMat
        new_x = np.full((1, no_nodes), inv_biomass)
        self.xMat = np.vstack([self.xMat, new_x])

        if trophLev == 0:
            # Producer: generate a new growth rate vector (ensure gen_r_vec returns correct length)
            new_r = self.gen_r_vec().reshape(1, no_nodes)
            self.rMat = np.vstack([self.rMat, new_r])
        else:
            # Consumer: start with zero growth rate row
            new_r = np.zeros((1, no_nodes))
            self.rMat = np.vstack([self.rMat, new_r])

        # Update cMat
        total_species_before = self.S_p + self.S_c
        if self.cMat.size == 0:
            # First species
            self.cMat = np.zeros((1,1))
        else:
            n_existing = self.cMat.shape[0]
            interaction_row = []
            for i in range(n_existing):
                if trophLev == 0:
                    # Invading producer
                    if i < self.S_p:
                        # Producer-Producer competition (negative)
                        interaction_coeff = np.random.uniform(-0.1,0.0)
                    else:
                        # Producer relative to existing consumers (neutral or slight)
                        interaction_coeff = np.random.uniform(-0.05,0.05)
                else:
                    # Invading consumer
                    if i < self.S_p:
                        # Consumer exploits producer (positive)
                        interaction_coeff = np.random.uniform(0.0, 0.1)
                    else:
                        # Consumer-Consumer competition (negative)
                        interaction_coeff = np.random.uniform(-0.1,0.0)
                interaction_row.append(interaction_coeff)
            interaction_row = np.array(interaction_row)

            self.cMat = np.vstack([self.cMat, interaction_row])
            new_col = np.zeros((self.cMat.shape[0],1))
            self.cMat = np.hstack([self.cMat, new_col])

        # Update S_p or S_c
        if trophLev == 0:
            self.S_p += 1
            logger.debug("Producer added to the community.")
        else:
            self.S_c += 1
            logger.debug("Consumer added to the community.")

        # Add a row to sMat for this new species
        s_add = np.random.uniform(0.1, 1.0, (1, no_nodes))
        self.sMat = np.vstack([self.sMat, s_add])

        # Update scVec and scVec_prime to match the new species count
        add_vec = np.ones((1, no_nodes))
        self.scVec = np.vstack([self.scVec, add_vec])

        add_prime = np.zeros((1, no_nodes))
        self.scVec_prime = np.vstack([self.scVec_prime, add_prime])

        # Add a row to tMat for this new species
        # For instance, temperature optimum = 0.5, scaling factor = 0.1
        t_add = np.array([[0.5, 0.1]])
        self.tMat = np.vstack([self.tMat, t_add])

        # Validate final dimensions
        total_species = self.S_p + self.S_c
        assert self.xMat.shape == (total_species, no_nodes)
        assert self.rMat.shape == (total_species, no_nodes)
        assert self.cMat.shape == (total_species, total_species), "cMat dimension mismatch"
        assert self.sMat.shape == (total_species, no_nodes)
        assert self.scVec.shape == (total_species, no_nodes)
        assert self.scVec_prime.shape == (total_species, no_nodes)
        assert self.tMat.shape == (total_species, 2), "tMat must have one row per species."

        logger.info(f"Species invaded - Trophic Level: {trophLev}, Producers: {self.S_p}, Consumers: {self.S_c}.")
        logger.debug(f"xMat shape: {self.xMat.shape}, rMat shape: {self.rMat.shape}, cMat shape: {self.cMat.shape}, tMat shape: {self.tMat.shape}")



    def extinct(self, wholeDom=True, ind_p=None, ind_c=None):
        """
        Remove extinct species from the community.

        Parameters:
        wholeDom (bool): Whether to check all species for extinction based on thresholds.
        ind_p (array-like, optional): Indices of producers to remove.
        ind_c (array-like, optional): Indices of consumers to remove.

        Returns:
        list: List containing arrays of removed producer and consumer indices.
        """
        if ind_p is None:
            ind_p = np.array([], dtype=int)
        if ind_c is None:
            ind_c = np.array([], dtype=int)

        indReturn = [None, None]  # Store extinction indices

        # Ensure 'self.thresh' is defined
        if not hasattr(self, 'thresh') or self.thresh is None:
            # self.thresh = 1e-3  # Define a default extinction threshold
            self.thresh = 5e-4  # Lowered extinction threshold
            logger.debug(f"Set default extinction threshold to {self.thresh}.")

        # Check for extinct producers
        if wholeDom:
            ind_p = []
            for i in range(self.S_p):
                if np.all(self.xMat[i, :] <= self.thresh):  # Check for extinction
                    ind_p.append(i)
            ind_p = np.array(ind_p, dtype=int)
        indReturn[0] = ind_p

        # Remove extinct producers
        if ind_p.size > 0:
            # Remove producers first to avoid shifting indices
            for i in reversed(ind_p):  # Iterate in reverse order to prevent index shifting
                logger.debug(f"Removing extinct producer at index {i}.")
                self.xMat = np.delete(self.xMat, i, axis=0)
                self.rMat = np.delete(self.rMat, i, axis=0)
                if hasattr(self, 'sMat') and self.sMat.shape[0] > 0:
                    self.sMat = np.delete(self.sMat, i, axis=0)
                if hasattr(self, 'tMat') and self.tMat.shape[0] > 0:
                    self.tMat = np.delete(self.tMat, i, axis=0)
                if self.emMat is not None and self.emMat.shape[0] > 0:
                    self.emMat = np.delete(self.emMat, i, axis=0)

                # Remove corresponding row and column from cMat
                if self.cMat.shape[0] > 0 and self.cMat.shape[1] > 0:
                    self.cMat = np.delete(self.cMat, i, axis=0)
                    self.cMat = np.delete(self.cMat, i, axis=1)
                    logger.debug(f"Updated cMat shape after removing producer: {self.cMat.shape}")

                # Update indices_S and indices_DP by removing i
                #  'indices_S' and 'indices_DP' are numpy arrays
                if self.indices_S is not None:
                    self.indices_S = self.indices_S[self.indices_S != i]
                    logger.debug(f"Updated indices_S: {self.indices_S}")
                if self.indices_DP is not None:
                    self.indices_DP = self.indices_DP[self.indices_DP != i]
                    logger.debug(f"Updated indices_DP: {self.indices_DP}")

            self.S_p -= len(ind_p)
            logger.debug(f"Removed {len(ind_p)} extinct producers from the community. New S_p: {self.S_p}")

        # Check for extinct consumers
        if wholeDom:
            ind_c = []
            for i in range(self.S_p, self.xMat.shape[0]):
                if np.all(self.xMat[i, :] <= self.thresh):
                    ind_c.append(i)
            ind_c = np.array(ind_c, dtype=int)
        indReturn[1] = ind_c

        # Remove extinct consumers
        if ind_c.size > 0:
            for i in reversed(ind_c):
                logger.debug(f"Removing extinct consumer at index {i}.")
                self.xMat = np.delete(self.xMat, i, axis=0)
                if self.cMat.shape[0] > 0 and self.cMat.shape[1] > 0:
                    self.cMat = np.delete(self.cMat, i, axis=0)
                    self.cMat = np.delete(self.cMat, i, axis=1)
                    logger.debug(f"Updated cMat shape after removing consumer: {self.cMat.shape}")
                if self.emMat is not None and self.emMat.shape[0] > 0:
                    self.emMat = np.delete(self.emMat, i, axis=0)

                # Update indices_S and indices_DP by removing i
                if self.indices_S is not None:
                    self.indices_S = self.indices_S[self.indices_S != i]
                    logger.debug(f"Updated indices_S: {self.indices_S}")
                if self.indices_DP is not None:
                    self.indices_DP = self.indices_DP[self.indices_DP != i]
                    logger.debug(f"Updated indices_DP: {self.indices_DP}")

            self.S_c -= len(ind_c)
            logger.debug(f"Removed {len(ind_c)} extinct consumers from the community. New S_c: {self.S_c}")

        # Update species richness
        if hasattr(self, 'sppRichness'):
            if self.sppRichness.shape[0] == 0:
                self.sppRichness = np.array([[self.S_p, self.S_c]])
                logger.debug("Initialized sppRichness.")
            else:
                new_row = np.array([[self.S_p, self.S_c]])
                self.sppRichness = np.vstack([self.sppRichness, new_row])
                logger.debug(f"Updated sppRichness: {self.sppRichness}")
        else:
            self.sppRichness = np.array([[self.S_p, self.S_c]])
            logger.debug("Initialized sppRichness.")

        return indReturn



    @staticmethod
    def _update_indices(indices, extinct_indices, S_before):
        indices = np.array(indices)
        extinct_indices = np.array(extinct_indices) 
        # Update indices for extinct species
        updated_indices = indices.copy()
        for j in range(len(extinct_indices) - 1):
            mask = (updated_indices > extinct_indices[j]) & (updated_indices < extinct_indices[j + 1])
            updated_indices[mask] -= (j + 1)
        return updated_indices


    def gen_disp_mat(self):
        """
        Generate the dispersal matrix with improved clarity, robustness, and validation.
        """
        logger.debug("Generating dispersal matrix.")

        if self.topo.no_nodes <= 1:
            # Handle single-node case
            self.dMat = np.zeros((1, 1))
            logger.debug("Dispersal matrix generated for a single node (no dispersal possible).")
            return

            
        if self.topo.adjMat is None:
            self.topo.gen_adj_mat()
            logger.debug("Adjacency matrix generated as it was not initialized.")

        # Compute dispersal matrix with exponential decay
        max_dist = np.max(self.topo.distMat)
        if max_dist == 0:
            logger.warning("All distances in distMat are zero. Check network generation.")
            self.dMat = np.zeros_like(self.topo.distMat)
            return

        self.dMat = np.exp(-self.topo.distMat / self.dispL)
        logger.debug(f"Initial dispersal matrix computed with dispersal length {self.dispL}.")

        # Apply adjacency constraints
        self.dMat *= self.topo.adjMat
        logger.debug("Dispersal matrix adjusted with adjacency matrix.")

        # Normalize dispersal rates
        kMat = np.zeros_like(self.dMat)
        for i in range(self.topo.no_nodes):
            neighbors = np.where(abs(self.topo.adjMat[:, i]) == 1)[0]
            num_neighbors = len(neighbors)
            if num_neighbors > 0:
                for j in neighbors:
                    if self.dispNorm == 0:  # Effort-weighted dispersal
                        if np.sum(self.dMat[:, i]) > 0:
                            kMat[i, j] = self.emRate / np.sum(self.dMat[:, i])
                    elif self.dispNorm == 1:  # Degree-weighted dispersal
                        kMat[i, j] = self.emRate / num_neighbors
                    elif self.dispNorm == 2:  # Passive dispersal
                        kMat[i, j] = self.emRate

        # Multiply with kMat to finalize dispersal matrix
        self.dMat *= kMat
        logger.debug(f"Dispersal matrix normalized using dispersal normalization method {self.dispNorm}.")

        # Include diagonal terms for self-dispersal (optional)
        if self.emRate < 0:
            np.fill_diagonal(self.dMat, -abs(self.emRate))
            logger.debug("Negative emigration rate applied to diagonal for self-dispersal.")

        # Final validation
        if not np.all(np.isfinite(self.dMat)):
            logger.error("Dispersal matrix contains invalid values (NaN or inf). Check parameters.")
            raise ValueError("Dispersal matrix contains invalid values.")

        logger.info("Dispersal matrix generated successfully.")
        logger.debug(f"Dispersal matrix: \n{self.dMat}")




    def log_state(self):
        """Log the current state of the species."""
        logger.debug(f"Producers: {self.S_p}, Consumers: {self.S_c}")
        logger.debug(f"xMat shape: {self.xMat.shape}")
        logger.debug(f"rMat shape: {self.rMat.shape}")
        logger.debug(f"cMat shape: {self.cMat.shape}")


            
# Example usage
# Corrected instantiation of Topography in species.py
if __name__ == "__main__":
    # Properly initialize the Topography class with all required arguments
    topo = Topography(
        no_nodes=100,
        lattice_height=10,
        lattice_width=10,
        phi=1.0,
        envVar=3,
        
        skVec=np.array([0.1, 0.2, 0.3]),  # skVec is still passed as a parameter if needed elsewhere
        var_e=1.0,
        randGraph=True,
        gabriel=True,
        T_int=25.0,
        network_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
        scVec=np.array([0.05]),
        consArea_bin=np.array([1 if i % 2 == 0 else 0 for i in range(100)]),  # Example binary conservation area
        consArea_multiplicative=True  # Conservation area perturbation mode
    )

    # Generate the network explicitly
    topo.gen_network()

    # Create an instance of Species using the Topography instance
    species = Species(topo)
    species.gen_disp_mat()  # Generate dispersal matrix
    species.invade(0)       # Invade with a producer
    species.invade(1)       # Invade with a consumer
    species.extinct()       # Remove extinct species
    species.log_state()     # Log the state of the Species








