# -*- coding: utf-8 -*-
"""CommunityDynamics.py

Updated and completed version to ensure it runs the model without missing parts.
"""

import numpy as np
import scipy.sparse as sp
from assimulo.problem import Explicit_Problem
import copy
from assimulo.solvers import CVode
print("Assimulo and CVode imported successfully.")
from scipy.spatial.distance import pdist, squareform
import logging

# Import necessary libraries
from lvmcm_rng import LVMCM_rng
# from metacommunity import Metacommunity


# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


import numpy as np
import logging
import scipy.sparse as sp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class CommunityDynamics:
    def __init__(
        self,
        spp,
        xMat,
        cMat,
        rMat,
        efMat=None,
        scVec=None,
        scVec_prime=None,
        rho=None,
        indices_S=None,
        indices_DP=None,
        bodymass=None,
        g_form_of_dynamics=False,
        emMat=None,
        dMat=None,
        bodymass_inv=None,
        mu=None
    ):
        # Assign parameters to member variables
        self.spp = spp
        self.xMat = xMat
        self.cMat = cMat
        self.rMat = rMat
        self.efMat = efMat if efMat is not None else np.zeros_like(rMat)
        
        # Ensure scVec is 2D with shape (1, nodes)
        if scVec is not None:
            if scVec.ndim == 1:
                self.scVec = scVec.reshape(1, -1)
                logger.debug(f"Reshaped scVec to {self.scVec.shape}")
            elif scVec.ndim == 2 and scVec.shape[0] == 1:
                self.scVec = scVec
            else:
                logger.warning(
                    f"scVec has shape {scVec.shape}, expected (1, {xMat.shape[1]}). Reshaping by averaging across species."
                )
                self.scVec = scVec.mean(axis=0, keepdims=True)
                logger.debug(f"Reshaped scVec to {self.scVec.shape} by averaging.")
        else:
            self.scVec = np.ones((1, xMat.shape[1]))
            logger.debug(f"Initialized scVec with shape {self.scVec.shape}")

        # Ensure scVec_prime is 2D with shape (1, nodes)
        if scVec_prime is not None:
            if scVec_prime.ndim == 1:
                self.scVec_prime = scVec_prime.reshape(1, -1)
                logger.debug(f"Reshaped scVec_prime to {self.scVec_prime.shape}")
            elif scVec_prime.ndim == 2 and scVec_prime.shape[0] == 1:
                self.scVec_prime = scVec_prime
            else:
                logger.warning(
                    f"scVec_prime has shape {scVec_prime.shape}, expected (1, {xMat.shape[1]}). Reshaping by averaging across species."
                )
                self.scVec_prime = scVec_prime.mean(axis=0, keepdims=True)
                logger.debug(f"Reshaped scVec_prime to {self.scVec_prime.shape} by averaging.")
        else:
            self.scVec_prime = np.ones((1, xMat.shape[1]))
            logger.debug(f"Initialized scVec_prime with shape {self.scVec_prime.shape}")
        
        self.rho = rho if rho is not None else 0.0
        self.indices_S = indices_S if indices_S is not None else np.arange(xMat.shape[0])
        self.indices_DP = indices_DP if indices_DP is not None else np.arange(xMat.size)
        self.bodymass = bodymass
        self.g_form_of_dynamics = g_form_of_dynamics
        self.emMat = emMat if emMat is not None else np.zeros_like(xMat)
        self.dMat = dMat
        self.dMat_sp = sp.csr_matrix(dMat) if dMat is not None else None  # Convert to sparse
        self.bodymass_inv = bodymass_inv if bodymass_inv is not None else 1.0
        self.mu = mu if mu is not None else 1.0

        # Initialize time tracking
        self.current_time = 0.0
        self.the_start_time = 0.0  # Initial start time for dynamics
        logger.debug("CommunityDynamics initialized with current_time = %s", self.current_time)
        if self.dMat_sp is not None:
            logger.debug("CommunityDynamics initialized with dMat_sp shape: %s", self.dMat_sp.shape)
        else:
            logger.error("Dispersal matrix (`dMat_sp`) is not initialized.")
            raise ValueError("The dispersal matrix (`dMat`) must be provided and non-empty.")
        
        # Validate matrices upon initialization
        self.validate_matrices()
        

    def validate_matrices(self):
        """
        Validate the shapes of critical matrices to ensure compatibility.
        """
        # Validate that critical matrices are provided and non-empty
        if self.rMat is None or self.rMat.shape[0] == 0:
            logger.error("Initialization error: rMat is empty or improperly initialized.")
            raise ValueError("The `rMat` matrix must be properly initialized and non-empty.")
        
        if self.xMat is None or self.xMat.shape[0] == 0:
            logger.error("Initialization error: xMat is empty or improperly initialized.")
            raise ValueError("The `xMat` matrix must be properly initialized and non-empty.")
        
        if self.cMat is None or self.cMat.shape[0] == 0:
            logger.warning("Initialization warning: cMat is empty, interactions will not be computed.")

        # Validate dMat_sp
        if self.dMat_sp is None or self.dMat_sp.size == 0:
            logger.error("Initialization error: Dispersal matrix (`dMat_sp`) must be provided and non-empty.")
            raise ValueError("The dispersal matrix (`dMat_sp`) must be provided and non-empty.")

        # Validate shapes of critical matrices
        if self.efMat.shape != self.rMat.shape:
            if self.efMat.size == 0:
                logger.warning(
                    f"efMat is empty. Setting efMat to zeros like rMat. rMat shape: {self.rMat.shape}"
                )
                self.efMat = np.zeros_like(self.rMat)
                logger.debug(f"Set efMat to zeros with shape: {self.efMat.shape}")
            else:
                logger.warning(
                    f"Reshaping efMat to match rMat. efMat shape: {self.efMat.shape}, rMat shape: {self.rMat.shape}"
                )
                if self.efMat.shape[0] != self.rMat.shape[0] or self.efMat.shape[1] != self.rMat.shape[1]:
                    logger.error(
                        "Cannot reshape `efMat` to match `rMat` due to incompatible dimensions."
                    )
                    raise ValueError("Incompatible dimensions between `efMat` and `rMat` for reshaping.")
                self.efMat = self.efMat.reshape(self.rMat.shape)
                logger.debug(f"Reshaped efMat to {self.efMat.shape}")
        
        # Additional validations and reshaping if necessary
        if self.emMat is None or self.emMat.size == 0:
            logger.warning("emMat is either None or empty. Dynamics may be incomplete.")
        if self.bodymass_inv is None:
            logger.warning("bodymass_inv was None. Initialized to default value of 1.0.")
        if self.cMat is not None and self.cMat.shape[0] != self.rMat.shape[0]:
            logger.warning(
                "Reshaping cMat to square matrix of rMat dimensions. cMat shape: %s", self.cMat.shape
            )
            self.cMat = np.zeros((self.rMat.shape[0], self.rMat.shape[0]))  # Fix cMat dimensions

        # Ensure scVec and scVec_prime are 2D with shape (1, nodes)
        for vec_name in ['scVec', 'scVec_prime']:
            vec = getattr(self, vec_name)
            if vec.shape != (1, self.xMat.shape[1]):
                logger.warning(
                    f"{vec_name} has shape {vec.shape}, expected (1, {self.xMat.shape[1]}). Reshaping to (1, {self.xMat.shape[1]})."
                )
                # Ensure it's (1, nodes)
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)
                elif vec.ndim == 2 and vec.shape[0] != 1:
                    vec = vec.mean(axis=0, keepdims=True)
                else:
                    vec = np.ones((1, self.xMat.shape[1]))
                setattr(self, vec_name, vec)
                logger.debug(f"Reshaped {vec_name} to {vec.shape} by averaging.")
        
        logger.debug(f"CommunityDynamics initialized with scVec shape: {self.scVec.shape}")
        logger.debug(f"CommunityDynamics initialized with scVec_prime shape: {self.scVec_prime.shape}")


    def number_of_variables(self):
        """
        Defines the number of state variables.
        
        Returns:
            int: Number of variables.
        """
        return self.xMat.shape[0] * self.xMat.shape[1]

    def read_state_from(self, state):
        """
        Converts 1D vector to N-dimensional biomass matrix.
        """
        if state.size != self.xMat.size:
            raise ValueError(f"State size mismatch: {state.size} does not match xMat size {self.xMat.size}")
        self.xMat = state.reshape(self.xMat.shape)

    def write_state_to(self):
        """
        Converts N-dimensional biomass matrix to 1D vector.
        """
        return self.xMat.flatten()



    def prepare_for_integration(self):
        """
        Prepares the dispersal matrix for integration.
        """
        if self.dMat is not None and self.dMat.size > 0:
            self.dMat_sp = sp.csr_matrix(self.dMat)
        else:
            logger.error("Error: `dMat` is not properly initialized before integration.")
            raise ValueError("Dispersal matrix (`dMat`) must be properly initialized before integration.")


    def number_of_root_functions(self):
        """
        Defines the number of root functions based on form of dynamics.
        
        Returns:
            int: Number of root functions.
        """
        if self.g_form_of_dynamics:
            return 2 * self.indices_S.shape[0] + self.indices_DP.shape[0]
        else:
            return 1

    def compute_intrinsic_growth_rates(self, B):
        """
        Computes the intrinsic growth rates.

        Args:
            B (ndarray): Biomass matrix (species x nodes).

        Returns:
            ndarray: Growth rates (species x nodes).
        """
        logger.debug("Starting computation of intrinsic growth rates.")
        
        # Validate `rMat` initialization
        if self.rMat.shape[0] == 0 or self.rMat.shape[1] == 0:
            logger.error("Error: `rMat` is improperly initialized or empty.")
            raise ValueError("The `rMat` matrix must be properly initialized with non-zero dimensions.")

        # Validate `cMat` initialization
        if self.cMat is None:
            logger.warning("`cMat` is None. Proceeding without interspecific interactions.")

        species, nodes = B.shape
        expected_species = self.rMat.shape[0]
        expected_nodes = self.rMat.shape[1] if self.rMat.ndim > 1 else 1

        #Validate the shape of B against `rMat`
        if species != expected_species:
            logger.error(f"Biomass matrix `B` has {species} species, expected {expected_species}.")
            raise ValueError("Number of species in `B` does not match `rMat`.")

        #Initialize `Gt` with a copy of `rMat`
        Gt = self.rMat.copy()  # Shape: (species, nodes)
        logger.debug(f"Initialized Gt with shape: {Gt.shape}")

        #Ensure `efMat` matches `rMat`; if not, initialize to zeros
        if self.efMat.shape != self.rMat.shape:
            if self.efMat.size == 0:
                logger.warning(f"`efMat` is empty. Initializing to zeros with shape {self.rMat.shape}.")
                self.efMat = np.zeros_like(self.rMat)
            else:
                logger.warning(f"Reshaping `efMat` to match `rMat`.")
                try:
                    self.efMat = self.efMat.reshape(self.rMat.shape)
                    logger.debug(f"Reshaped `efMat` to {self.efMat.shape}.")
                except ValueError:
                    logger.error("Cannot reshape `efMat` to match `rMat` due to incompatible dimensions.")
                    raise ValueError("Incompatible dimensions between `efMat` and `rMat` for reshaping.")

        #Handle competition and interaction terms if `cMat` is defined
        if self.cMat is not None:
            try:
                #Ensure `cMat` is square and matches species count
                if self.cMat.shape != (species, species):
                    logger.warning(
                        f"Reshaping `cMat` from {self.cMat.shape} to ({species}, {species}). Initializing to zeros."
                    )
                    self.cMat = np.zeros((species, species))
                    logger.debug(f"Reshaped `cMat` to {self.cMat.shape}.")

                #Compute inter-specific interactions
                intersp = self.cMat @ B  # Shape: (species, nodes)
                logger.debug(f"Computed inter-specific interactions with shape: {intersp.shape}")

                # Scale inter-specific interactions using `scVec`
                if self.scVec is not None:
                    # Ensure `scVec` is (1, nodes)
                    if self.scVec.shape != (1, nodes):
                        logger.warning(
                            f"`scVec` has shape {self.scVec.shape}, expected (1, {nodes}). Reshaping by averaging."
                        )
                        if self.scVec.ndim == 1:
                            self.scVec = self.scVec.reshape(1, -1)
                            logger.debug(f"Reshaped `scVec` to {self.scVec.shape}.")
                        elif self.scVec.ndim == 2 and self.scVec.shape[0] != 1:
                            self.scVec = self.scVec.mean(axis=0, keepdims=True)
                            logger.debug(f"Reshaped `scVec` to {self.scVec.shape}.")
                        else:
                            logger.warning(f"Cannot reshape `scVec` properly. Initializing to ones.")
                            self.scVec = np.ones((1, nodes))
                            logger.debug(f"Initialized `scVec` to ones with shape: {self.scVec.shape}.")
                    intersp *= self.scVec  # Broadcasting (1, nodes) to (species, nodes)
                    logger.debug("Applied `scVec` scaling to inter-specific interactions.")
                else:
                    logger.warning("`scVec` is None. Skipping scaling of inter-specific interactions.")

                # Compute intra-specific interactions using `scVec_prime`
                if self.scVec_prime is not None:
                    # Ensure `scVec_prime` is (1, nodes)
                    if self.scVec_prime.shape != (1, nodes):
                        logger.warning(
                            f"`scVec_prime` has shape {self.scVec_prime.shape}, expected (1, {nodes}). Reshaping by averaging."
                        )
                        if self.scVec_prime.ndim == 1:
                            self.scVec_prime = self.scVec_prime.reshape(1, -1)
                            logger.debug(f"Reshaped `scVec_prime` to {self.scVec_prime.shape}.")
                        elif self.scVec_prime.ndim == 2 and self.scVec_prime.shape[0] != 1:
                            self.scVec_prime = self.scVec_prime.mean(axis=0, keepdims=True)
                            logger.debug(f"Reshaped `scVec_prime` to {self.scVec_prime.shape}.")
                        else:
                            logger.warning(f"Cannot reshape `scVec_prime` properly. Initializing to ones.")
                            self.scVec_prime = np.ones((1, nodes))
                            logger.debug(f"Initialized `scVec_prime` to ones with shape: {self.scVec_prime.shape}.")
                    intrasp = B * self.scVec_prime  # Element-wise multiplication, broadcasting
                    logger.debug(f"Computed intra-specific interactions with shape: {intrasp.shape}")
                    Gt -= (intersp + intrasp)
                    logger.debug("Subtracted inter-specific and intra-specific interactions from `Gt`.")
                else:
                    Gt -= intersp
                    logger.debug("Subtracted inter-specific interactions from `Gt` (no intra-specific interactions).")

            except ValueError as e:
                logger.error(f"Error computing competition interactions: {e}")
                raise

        else:
            # No competition matrix; assume simple logistic-like interaction
            try:
                Gt -= B  # Shape: (species, nodes)
                logger.debug("Subtracted biomass directly from `Gt` (no competition matrix).")
            except ValueError as e:
                logger.error(f"Error in logistic interaction computation: {e}")
                raise

        #  Handle trophic interactions and spatial scaling
        if species > self.rMat.shape[0]:
            try:
                # Define the number of producers and consumers
                num_producers = self.rMat.shape[0]
                num_consumers = species - num_producers

                # Validate `cMat` dimensions for trophic interactions
                if self.cMat.shape != (species, species):
                    logger.warning(
                        f"`cMat` shape {self.cMat.shape} does not match expected ({species}, {species}) for trophic interactions."
                    )
                    self.cMat = np.zeros((species, species))
                    logger.debug(f"Reinitialized `cMat` to zeros with shape {self.cMat.shape}.")

                # Consumer to producer interactions
                consumer_interactions = (
                    self.cMat[:num_producers, num_producers:] @ B[num_producers:, :]
                )  # Shape: (num_producers, nodes)
                logger.debug(f"Computed consumer interactions with shape: {consumer_interactions.shape}")

                Gt[:num_producers, :] -= consumer_interactions
                logger.debug("Subtracted consumer interactions from producers in `Gt`.")

                # Producer to consumer interactions
                producer_interactions = (
                    self.cMat[num_producers:, :num_producers] @ B[:num_producers, :]
                )  # Shape: (num_consumers, nodes)
                logger.debug(f"Computed producer interactions with shape: {producer_interactions.shape}")

                Gt[num_producers:, :] = self.rho * (producer_interactions - 1)
                logger.debug("Applied producer interactions to consumers in `Gt`.")

            except ValueError as e:
                logger.error(f"Error in trophic interactions or spatial scaling: {e}")
                raise

        #  validation before returning
        if not np.all(np.isfinite(Gt)):
            logger.error("Computed growth rates contain non-finite values.")
            raise ValueError("Growth rates contain non-finite values.")

        logger.debug("Intrinsic growth rates computed successfully.")
        return Gt  # Shape: (species, nodes)


    def dynamics(self, t, state):
        """
        Computes the dynamics of the system.

        Args:
            t (float): Current time.
            state (ndarray): Current state of the system as a 1D numpy array.

        Returns:
            ndarray: Time derivative of the state.
        """
        logger.debug(f"Starting dynamics computation at time {t}.")
        
        # Validate state size
        if state.size != self.xMat.size:
            logger.error(f"State size {state.size} does not match xMat size {self.xMat.size}.")
            raise ValueError(f"State size {state.size} must match xMat size {self.xMat.size}.")

        # Reshape state to biomass matrix
        try:
            X = state.reshape(self.xMat.shape)
            logger.debug("State reshaped successfully to match xMat.")
        except ValueError as e:
            logger.error(f"Error reshaping state elements: {e}")
            raise ValueError("Failed to reshape state elements to match xMat.") from e

        # Initialize biomass matrix B
        B = np.zeros_like(self.xMat)
        B.flat[self.indices_DP] = X.flat[self.indices_DP]
        logger.debug(f"Biomass matrix B initialized with shape {B.shape}.")

        # Compute intrinsic growth rates
        try:
            Gt = self.compute_intrinsic_growth_rates(B)
        except ValueError as e:
            logger.error("Error in computing intrinsic growth rates.")
            raise

        # Compute derivative dXdt
        dXdt = Gt.copy()  # Shape: (species, nodes)
        logger.debug(f"dXdt initialized with shape: {dXdt.shape}")

        # Apply dynamic populations scaling if necessary
        dXdt.flat[self.indices_DP] *= B.flat[self.indices_DP]
        logger.debug("Applied dynamic populations scaling to dXdt.")

        # Handle mass effect (dispersal)
        massEffect = None
        if self.emMat is None:
            if self.dMat_sp.shape[0] > 0:
                if B.shape[1] != self.dMat_sp.shape[0]:
                    logger.error(
                        f"Shape mismatch: B shape = {B.shape}, dMat_sp shape = {self.dMat_sp.shape}"
                    )
                    raise ValueError("Biomass matrix `B` and dispersal matrix `dMat_sp` must have compatible dimensions.")
                massEffect = B @ self.dMat_sp  # Shape: (species, nodes) @ (nodes, nodes) = (species, nodes)
                dXdt += massEffect
                logger.debug("Mass effect calculated without environmental matrix (emMat).")
            else:
                logger.error("Error: `dMat_sp` is improperly initialized for mass effect calculation.")
                raise ValueError("Dispersal matrix (`dMat_sp`) is improperly initialized.")
        else:
            try:
                if self.emMat.ndim == 1:
                    emMat_N = np.repeat(self.emMat[:, np.newaxis], B.shape[1], axis=1)  # Shape: (species, nodes)
                elif self.emMat.shape[0] == B.shape[0]:
                    emMat_N = self.emMat
                else:
                    logger.error(
                        f"Error reshaping emMat: shape {self.emMat.shape} cannot align with B shape {B.shape}"
                    )
                    raise ValueError("emMat dimensions are incompatible with B.")
                logger.debug(f"emMat reshaped to align with B: shape {emMat_N.shape}")

                if self.dMat_sp.shape[0] > 0:
                    if (emMat_N * B).shape[1] != self.dMat_sp.shape[0]:
                        logger.error(
                            f"Shape mismatch for mass effect computation: (emMat_N * B) shape = {(emMat_N * B).shape}, dMat_sp shape = {self.dMat_sp.shape}"
                        )
                        raise ValueError("Incompatible dimensions for mass effect computation.")
                    massEffect = (emMat_N * B) @ self.dMat_sp  # Shape: (species, nodes) @ (nodes, nodes) = (species, nodes)
                    dXdt += massEffect
                    logger.debug("Mass effect calculated with environmental matrix (emMat).")
                else:
                    logger.error("Error: `dMat_sp` is improperly initialized for mass effect calculation with emMat.")
                    raise ValueError("Dispersal matrix (`dMat_sp`) is improperly initialized.")
            except Exception as e:
                logger.error(f"Error reshaping emMat: {e}")
                raise ValueError("Error reshaping emMat to align with B.") from e

        # Handle consumer biomass changes
        if massEffect is not None:
            try:
                # Avoid division by zero by adding a small epsilon
                epsilon = 1e-8
                denominator = Gt.flat[self.indices_S] + self.mu  # Shape: (len(indices_S),)
                denominator = np.where(denominator == 0, epsilon, denominator)
                logger.debug(f"Denominator for consumer biomass changes: {denominator}")

                valid_indices = np.where(denominator != 0)[0]
                dPsi = np.zeros_like(Gt.flat[self.indices_S])
                dPsi[valid_indices] = (
                    self.bodymass_inv
                    * massEffect.flat[self.indices_S][valid_indices]
                    * Gt.flat[self.indices_S][valid_indices]
                ) / denominator[valid_indices]
                dPsi[Gt.flat[self.indices_S] < 0] = 0
                dXdt.flat[self.indices_S] = dPsi
                logger.debug("Consumer biomass changes computed successfully.")
            except Exception as e:
                logger.error("Error in computing consumer biomass changes.")
                raise

        # Handle negative biomass values
        negativeB = np.where(B < 0)
        if len(negativeB[0]) > 0:
            logger.warning(f"Negative biomass values found at indices: {negativeB}")
            dXdt[negativeB] = -B[negativeB]

        # Replace non-finite values
        nonfiniteX = np.where(~np.isfinite(dXdt))
        if len(nonfiniteX[0]) > 0:
            logger.warning(f"Non-finite values in dXdt at indices: {nonfiniteX}")
            dXdt[nonfiniteX] = 0

        # Check for blow-up values in dXdt
        blowupX = np.where(np.abs(dXdt) > 1e10)
        if len(blowupX[0]) > 0:
            logger.warning("Warning: dXdt blow-up detected. Setting these values to zero.")
            dXdt[blowupX] = 0

        logger.debug("Dynamics computation completed successfully.")
        return dXdt.flatten()


    def root_functions(self, t, state):
        """
        Defines the root functions to be used during event detection.

        Args:
            t (float): Current time.
            state (ndarray): Current state of the system as a 1D numpy array.

        Returns:
            ndarray: Computed root values.
        """
        try:
            logger.info("Evaluating root functions...")

            # Root functions to find events, such as crossing specific thresholds
            root_values = state.copy()  # Each state value potentially represents a root event if it crosses zero
            
            # Adjust root_values to ensure it returns a single scalar per event for Assimulo compatibility
            root_values = np.min(state)  # Find the minimum value to detect when the state crosses zero

            logger.debug(f"Root functions computed: {root_values}")
            return root_values
        except Exception as e:
            logger.error(f"Error in root function computation: {e}")
            raise

    def react_to_roots(self, root_indices):
        """
        React to detected root events by updating system states.

        Args:
            root_indices (list): Indices of root events that were detected.
        """
        try:
            logger.info("Reacting to root events...")

            if not root_indices:
                logger.debug("No root events to react to.")
                return

            indices_DP_tmp = list(self.indices_DP) if self.indices_DP is not None else []
            indices_S_tmp = list(self.indices_S) if self.indices_S is not None else []

            root_indices_vec = np.abs(np.array(root_indices, dtype=int)) - 1  # Adjust indexing to start from zero

            for i in reversed(root_indices_vec):
                if i < len(indices_S_tmp):
                    # Detected change in state of Poisson clock, S->D transition
                    logger.info("S->D detected")
                    true_1Dindex = indices_S_tmp[i]

                    # Compute establishment probability and update state
                    growthRate_i = self.rMat[true_1Dindex]
                    density_dependence = np.dot(self.cMat[true_1Dindex], self.xMat[:, true_1Dindex])
                    growthRate_i -= density_dependence
                    probEstablish = growthRate_i / (growthRate_i - self.mu) if growthRate_i != self.mu else 1.0
                    self.xMat[true_1Dindex] = self.bodymass / probEstablish

                    # Update indices
                    del indices_S_tmp[i]
                    indices_DP_tmp.append(true_1Dindex)
                elif i < len(indices_S_tmp) * 2:
                    # Handle S->P transition
                    logger.info("S->P detected")
                    true_1Dindex = indices_S_tmp[i - len(indices_S_tmp)]
                    if root_indices[i] > 0:
                        continue
                else:
                    # Handle P->D/S transition
                    if i >= len(indices_S_tmp) * 2:
                        logger.warning(f"Index {i} out of bounds for root processing.")
                        continue
                    
                    logger.info("P->D/S detected")
                    true_1Dindex = indices_DP_tmp[i - 2 * len(indices_S_tmp)]
                    true_2Dindex = divmod(true_1Dindex, self.xMat.shape[0])

                    # Calculate probability of remaining extant
                    if self.dMat_sp is not None and self.bodymass_inv is not None:
                        massEffect = np.dot(self.xMat[true_2Dindex[0], :], self.dMat_sp[:, true_2Dindex[1]])
                        if massEffect.size > 0:
                            p_extant = 1 - np.power(2, -massEffect * self.bodymass_inv / self.mu)
                        else:
                            p_extant = 0
                    else:
                        p_extant = 0

                    z_PD = np.random.rand()
                    if z_PD < p_extant:
                        # Update biomass if species remains extant
                        self.xMat.flat[true_1Dindex] = max(self.bodymass, min(1.0, abs(self.xMat.flat[true_1Dindex]) / p_extant))
                    else:
                        # Set Poisson clock for state transition
                        poissonClock = np.random.rand()
                        poissonClock = np.log(poissonClock) - 1
                        self.xMat.flat[true_1Dindex] = poissonClock

                        # Update indices for DP and S compartments
                        if i - 2 * len(indices_S_tmp) < len(indices_DP_tmp):
                            del indices_DP_tmp[i - 2 * len(indices_S_tmp)]
                        indices_S_tmp.append(true_1Dindex)

            # Update the DP and S indices with the new values
            self.indices_DP = sorted(indices_DP_tmp) if len(indices_DP_tmp) > 0 else []
            self.indices_S = sorted(indices_S_tmp) if len(indices_S_tmp) > 0 else []

        except IndexError as e:
            logger.error(f"Index out of range in reacting to roots: {e}")
        except Exception as e:
            logger.error(f"Error in reacting to roots: {e}")
            raise



    @staticmethod
    def oneDto2Dind(oneDindex, nRows):
        """
        Converts 1D columnwise indices to 2D row-column indices.
        
        Args:
            oneDindex (int): 1D index.
            nRows (int): Number of rows in the 2D representation.
        
        Returns:
            tuple: Corresponding row and column indices.
        """
        row = oneDindex % nRows
        col = oneDindex // nRows
        return row, col


#  Example usage to test the CommunityDynamics class with the Metacommunity integration
def example_usage():
    """
    Example usage to test the CommunityDynamics class.
    """
    spp = None
    xMat = np.array([[0.5, 0.3], [0.2, 0.4]])
    cMat = np.random.rand(xMat.shape[0], xMat.shape[1])
    rMat = np.random.rand(xMat.shape[0], xMat.shape[1])
    efMat = np.random.rand(xMat.shape[0], xMat.shape[1])
    scVec = np.ones(xMat.shape[1])
    scVec_prime = np.ones(xMat.shape[1])
    dMat = np.random.rand(xMat.shape[0], xMat.shape[0])
    bodymass = 0.4
    bodymass_inv = 1 / bodymass
    mu = 0.2

    community_dynamics = CommunityDynamics(
        spp, xMat, cMat, rMat, efMat, scVec, scVec_prime, rho=0.2,
        bodymass=bodymass, g_form_of_dynamics=False, emMat=None,
        dMat=dMat, bodymass_inv=bodymass_inv, mu=mu
    )

    initial_state = community_dynamics.write_state_to()

    problem = Explicit_Problem(community_dynamics.dynamics, initial_state)
    solver = CVode(problem)

    # Perform the integration
    t, y = solver.simulate(tfinal=100)

    print("Integration completed successfully.")
    print("State values at the final time step:")
    print(y[-1].reshape(xMat.shape))

if __name__ == "__main__":
    example_usage()










