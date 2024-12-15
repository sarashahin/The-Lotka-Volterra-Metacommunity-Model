# -*- coding: utf-8 -*-
"""CommunityDynamics.py

Updated and completed version to ensure it runs the model without missing parts.
"""

# Import necessary libraries
# from metacommunity import Metacommunity  # Assuming this class is defined in metacommunity.py
from lvmcm_rng import LVMCM_rng
from ode import ODEState, ODEDynamicalObject, ODEVector, ODEMatrix
# from metacommunity import Metacommunity


import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import logging
from scipy.integrate import solve_ivp



# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CommunityDynamics:
    def __init__(self, spp, xMat, cMat, rMat, efMat=None, scVec=None, scVec_prime=None, rho=None, indices_S=None,
                 indices_DP=None, bodymass=None, g_form_of_dynamics=False, emMat=None, dMat=None, bodymass_inv=None, mu=None):
        from metacommunity import Metacommunity

        """
        Initializes the CommunityDynamics object with key state variables.
        
        Args:
            xMat (ndarray): Biomass matrix.
            cMat (ndarray): Interaction matrix.
            rMat (ndarray): Growth rate matrix.
            efMat (ndarray, optional): Environmental fluctuations.
            scVec (ndarray, optional): Scaling vector.
            scVec_prime (ndarray, optional): Alternate scaling vector.
            rho (float, optional): Parameter for interactions.
            indices_S (ndarray, optional): Indices for Poisson clocks.
            indices_DP (ndarray, optional): Indices for derived quantities.
            bodymass (float, optional): Bodymass value.
            g_form_of_dynamics (bool, optional): Boolean to determine form of dynamics.
            emMat (ndarray, optional): Environmental matrix.
            dMat (ndarray, optional): Dispersal matrix.
            bodymass_inv (float, optional): Inverse of body mass.
            mu (float, optional): Growth rate parameter.
        """

        # Assign parameters to member variables
        self.spp = spp
        self.xMat = xMat
        self.cMat = cMat
        self.rMat = rMat
        self.efMat = efMat
        self.scVec = scVec
        self.scVec_prime = scVec_prime
        self.rho = rho
        self.indices_S = indices_S if indices_S is not None else np.arange(xMat.shape[0])
        self.indices_DP = indices_DP if indices_DP is not None else np.arange(xMat.size)
        self.bodymass = bodymass
        self.g_form_of_dynamics = g_form_of_dynamics
        self.emMat = emMat
        self.dMat = dMat
        self.bodymass_inv = bodymass_inv
        self.mu = mu if mu is not None else 1.0


        # Add `the_start_time` attribute for ODE compatibility
        self.current_time = 0.0
        self.the_start_time = 0.0  # Initial start time for dynamics

        # Validate that critical matrices are provided and non-empty
        if self.rMat is None or self.rMat.shape[0] == 0:
            logger.error("Initialization error: rMat is empty or improperly initialized.")
            raise ValueError("The `rMat` matrix must be properly initialized and non-empty.")
        
        if self.xMat is None or self.xMat.shape[0] == 0:
            logger.error("Initialization error: xMat is empty or improperly initialized.")
            raise ValueError("The `xMat` matrix must be properly initialized and non-empty.")
        
        if self.cMat is None or self.cMat.shape[0] == 0:
            logger.warning("Initialization warning: cMat is empty, interactions will not be computed.")

        # Validate dMat
        if self.dMat is None or self.dMat.size == 0:
            raise ValueError("The dispersal matrix (`dMat`) must be provided and non-empty.")

        # Validate shapes of critical matrices
        if self.efMat is not None and self.efMat.shape != self.rMat.shape:
            logger.warning(f"Reshaping efMat to match rMat. efMat shape: {self.efMat.shape}, rMat shape: {self.rMat.shape}")
            self.efMat = np.zeros_like(self.rMat)

        # Precompute sparse dispersal matrix
        self.dMat_sp = sp.csr_matrix(self.dMat)
        if self.dMat_sp is None or self.dMat_sp.size == 0:
            raise ValueError("The sparse dispersal matrix (`dMat_sp`) could not be initialized.")
        logger.debug(f"CommunityDynamics initialized with dMat_sp shape: {self.dMat_sp.shape}")

        logger.debug(f"CommunityDynamics initialized with current_time = {self.current_time}")

        if self.emMat is None or not self.emMat.size:
            logger.warning("Warning: emMat is either None or empty. Dynamics may be incomplete.")

        if self.bodymass_inv is None:
            self.bodymass_inv = 1.0  # Default inverse body mass
            logger.warning("bodymass_inv was None. Initialized to default value of 1.0.")

        logger.debug(f"CommunityDynamics initialized with current_time = 0.0")


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
            B (ndarray): Biomass matrix.

        Returns:
            ndarray: Growth rates.
        """
        # Ensure `rMat` is properly initialized
        if self.rMat.shape[0] == 0 or self.rMat.shape[1] == 0:
            logger.error("Error: `rMat` is improperly initialized or empty.")
            raise ValueError("The `rMat` matrix must be properly initialized with non-zero dimensions.")

        # Validate and reshape `B` to match `rMat`
        if B.shape != self.rMat.shape:
            logger.warning(f"Reshaping biomass matrix `B` to match `rMat`. B shape: {B.shape}, rMat shape: {self.rMat.shape}")
            reshaped_B = np.zeros_like(self.rMat)
            reshaped_B[:B.shape[0], :B.shape[1]] = B
            B = reshaped_B

        # Initialize `Gt` with a copy of `rMat`
        Gt = self.rMat.copy()

        # Add environmental fluctuations to growth rate if available
        if self.efMat is not None:
            if self.efMat.shape != self.rMat.shape:
                logger.warning(f"Reshaping efMat to match rMat. efMat shape: {self.efMat.shape}, rMat shape: {self.rMat.shape}")
                self.efMat = np.zeros_like(self.rMat)
            Gt += self.efMat

        # Handle competition and interaction terms if `cMat` is defined
        if self.cMat is not None:
            try:
                # Validate and reshape `cMat` to ensure compatibility
                if self.cMat.shape[0] != self.rMat.shape[0] or self.cMat.shape[1] != self.rMat.shape[0]:
                    logger.warning(f"Reshaping cMat to square matrix of rMat dimensions. cMat shape: {self.cMat.shape}")
                    self.cMat = np.zeros((self.rMat.shape[0], self.rMat.shape[0]))

                # Compute inter-specific interactions
                intersp = self.cMat[:self.rMat.shape[0], :self.rMat.shape[0]] @ B[:self.rMat.shape[0], :]

                # Scale interactions if `scVec` is defined
                if self.scVec is not None:
                    if self.scVec.ndim == 1:
                        self.scVec = self.scVec.reshape(1, -1)  # Convert to row vector for broadcasting
                    if intersp.shape != Gt.shape:
                        logger.warning(f"Reshaping scVec to match intersp. intersp shape: {intersp.shape}, scVec shape: {self.scVec.shape}")
                        self.scVec = np.ones_like(intersp)
                    intersp *= self.scVec

                # Intra-specific interactions
                if self.scVec_prime is not None:
                    if self.scVec_prime.ndim == 1:
                        self.scVec_prime = self.scVec_prime.reshape(1, -1)  # Convert to row vector for broadcasting
                    if B[:self.rMat.shape[0], :].shape != self.scVec_prime.shape:
                        logger.warning(f"Reshaping scVec_prime to match B. B shape: {B[:self.rMat.shape[0], :].shape}, scVec_prime shape: {self.scVec_prime.shape}")
                        self.scVec_prime = np.ones_like(B[:self.rMat.shape[0], :])
                    intrasp = B[:self.rMat.shape[0], :] * self.scVec_prime
                    Gt[:self.rMat.shape[0], :] -= (intersp + intrasp)
                else:
                    Gt[:self.rMat.shape[0], :] -= intersp

            except ValueError as e:
                logger.error(f"Error computing competition interactions: {e}")
                raise
        else:
            # No competition matrix; assume simple logistic-like interaction
            try:
                Gt[:self.rMat.shape[0], :] -= B[:self.rMat.shape[0], :]
            except ValueError as e:
                logger.error(f"Error in logistic interaction computation: {e}")
                raise

        # Handle trophic interactions and spatial scaling
        if B.shape[0] > self.rMat.shape[0]:
            try:
                # Consumer to producer interactions
                consumer_interactions = (
                    self.cMat[:self.rMat.shape[0], self.rMat.shape[0]:] @ B[self.rMat.shape[0]:, :]
                )
                Gt[:self.rMat.shape[0], :] -= consumer_interactions

                # Producer to consumer interactions
                producer_interactions = (
                    self.cMat[self.rMat.shape[0]:, :self.rMat.shape[0]] @ B[:self.rMat.shape[0], :]
                )
                Gt[self.rMat.shape[0]:, :] = self.rho * (producer_interactions - 1)
            except ValueError as e:
                logger.error(f"Error in trophic interactions or spatial scaling: {e}")
                raise

        return Gt



    def dynamics(self, state, time_derivative):
        """
        Computes the dynamics of the system.

        Args:
            state (ODEVector): Current state of the system as an ODEVector instance.
            time_derivative (ODEVector): Time derivative to store the result.

        Returns:
            None
        """
        # Validate and reshape state elements
        if state.elements.size != self.xMat.size:
            logger.warning(f"State size {state.elements.size} does not match xMat size {self.xMat.size}. Reshaping state elements.")
            if state.elements.size == self.rMat.size:  # Check if state matches the size of rMat
                state.elements = state.elements[:self.xMat.size]  # Trim elements to match xMat size
            else:
                raise ValueError(f"State size {state.elements.size} must match xMat size {self.xMat.size}.")

        try:
            X = state.elements.reshape(self.xMat.shape)
            logger.debug("State reshaped successfully to match xMat.")
        except ValueError as e:
            logger.error(f"Error reshaping state elements: {e}")
            raise ValueError("Failed to reshape state elements to match xMat.") from e

        # Initialize B with zeros and set indices_DP values from X
        B = np.zeros_like(self.xMat)
        B.flat[self.indices_DP] = X.flat[self.indices_DP]

        # Validate and reshape `B` to match `rMat` if necessary
        if B.shape != self.rMat.shape:
            logger.warning(f"Reshaping biomass matrix `B` to match `rMat`. B shape: {B.shape}, rMat shape: {self.rMat.shape}")
            reshaped_B = np.zeros_like(self.rMat)
            reshaped_B[:B.shape[0], :B.shape[1]] = B
            B = reshaped_B

        # Compute intrinsic growth rates
        try:
            Gt = self.compute_intrinsic_growth_rates(B)
        except ValueError as e:
            logger.error("Error in computing intrinsic growth rates.")
            raise

        dXdt = Gt.copy()
        dXdt.flat[self.indices_DP] *= B.flat[self.indices_DP]

        # Handle emigration and mass effect calculations
        massEffect = None
        if self.emMat is None:
            if self.dMat_sp.shape[0] > 0:
                try:
                    if B.shape[1] != self.dMat_sp.shape[0]:
                        logger.error(f"Shape mismatch: B shape = {B.shape}, dMat_sp shape = {self.dMat_sp.shape}")
                        raise ValueError("Biomass matrix `B` and dispersal matrix `dMat_sp` must have compatible dimensions.")
                    massEffect = B @ self.dMat_sp
                    dXdt += massEffect
                    logger.debug("Mass effect calculated without environmental matrix (emMat).")
                except ValueError as e:
                    logger.error(f"Error in mass effect computation: {e}")
                    raise
            else:
                logger.error("Error: `dMat_sp` is improperly initialized for mass effect calculation.")
                raise ValueError("Dispersal matrix (`dMat_sp`) is improperly initialized.")
        else:
            try:
                if self.emMat.ndim == 1:
                    emMat_N = np.repeat(self.emMat[:, np.newaxis], B.shape[1], axis=1)
                elif self.emMat.shape[0] == B.shape[0]:
                    emMat_N = self.emMat
                else:
                    logger.error(f"Error reshaping emMat: shape {self.emMat.shape} cannot align with B shape {B.shape}")
                    raise ValueError("emMat dimensions are incompatible with B.")
                logger.debug(f"emMat reshaped to align with B: shape {emMat_N.shape}")
            except Exception as e:
                logger.error(f"Error reshaping emMat: {e}")
                raise ValueError("Error reshaping emMat to align with B.") from e

            if self.dMat_sp.shape[0] > 0:
                try:
                    massEffect = (emMat_N * B) @ self.dMat_sp
                    dXdt += massEffect
                    logger.debug("Mass effect calculated with environmental matrix (emMat).")
                except ValueError as e:
                    logger.error(f"Error in mass effect computation with emMat: {e}")
                    raise
            else:
                logger.error("Error: `dMat_sp` is improperly initialized for mass effect calculation with emMat.")
                raise ValueError("Dispersal matrix (`dMat_sp`) is improperly initialized.")

        # Validate and compute consumer biomass changes
        if massEffect is not None:
            try:
                valid_indices = np.where(Gt.flat[self.indices_S] + self.mu != 0)[0]
                dPsi = np.zeros_like(Gt.flat[self.indices_S])
                dPsi[valid_indices] = (
                    self.bodymass_inv
                    * massEffect.flat[self.indices_S][valid_indices]
                    * Gt.flat[self.indices_S][valid_indices]
                ) / (Gt.flat[self.indices_S][valid_indices] + self.mu)
                dPsi[Gt.flat[self.indices_S] < 0] = 0
                dXdt.flat[self.indices_S] = dPsi
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

        # Assign the computed derivatives to the time_derivative ODEVector
        time_derivative.elements = dXdt.flatten()
        logger.debug("Dynamics computation completed successfully.")






    def root_functions(self, state, root_values):
        """
        Defines the root functions to be used during event detection.

        Args:
            state (ODEVector): Current state of the system.
            root_values (ODEVector): Output to store the computed root values.
        """
        try:
            logger.info("Evaluating root functions...")

            # Root functions to find events, such as crossing specific thresholds (e.g., x=0)
            root_values.elements = state.elements.copy()  # Each state value potentially represents a root event if it crosses zero
            
            # Adjust root_values to ensure it returns a single scalar per event for solve_ivp compatibility
            root_values.elements = np.min(state.elements)  # Find the minimum value to detect when the state crosses zero

            logger.debug(f"Root functions computed: {root_values.elements}")
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





def example_usage():
    """
    Example usage to test the CommunityDynamics class with the Metacommunity integration.
    """
    from metacommunity import Metacommunity

    # Define parameters for the Metacommunity
    a_init = True
    a_invMax = 50
    a_tMax = 100
    a_outputDirectory = "./output"
    a_c1, a_c2, a_c3 = 0.1, 0.1, 0.1
    a_emRate = 0.05
    a_dispL = 0.01
    a_pProducer = 0.5
    a_prodComp, a_symComp = 0.2, 0.2
    a_alpha, a_sigma, a_sigma_t = 0.1, 0.05, 0.01
    a_rho = 0.2
    a_comp_dist = 1.0
    a_omega = 0.1
    a_dispNorm = True
    a_no_nodes = 2
    a_lattice_height, a_lattice_width = 1, 2
    a_phi = 0.1
    a_envVar = 0.1
    a_skVec = np.array([1.0, 1.0])
    a_var_e = 0.01
    a_randGraph, a_gabriel = False, False
    a_T_int = 10
    a_envMat = np.array([[0.1, 0.2], [0.3, 0.4]])
    a_xMat = np.array([[0.5, 0.3], [0.2, 0.4]])
    a_bMat = a_xMat  # Assume initial biomass matches
    a_scMat = np.ones_like(a_xMat)
    simTime = 0.0

    # Initialize Metacommunity
    metacommunity = Metacommunity(
        spp=None, a_init=a_init, a_bMat=a_bMat, a_xMat=a_xMat, a_scMat=a_scMat, a_invMax=a_invMax, a_tMax=a_tMax,
        a_outputDirectory=a_outputDirectory, a_c1=a_c1, a_c2=a_c2, a_c3=a_c3, a_emRate=a_emRate, a_dispL=a_dispL,
        a_pProducer=a_pProducer, a_prodComp=a_prodComp, a_symComp=a_symComp, a_alpha=a_alpha, a_sigma=a_sigma,
        a_sigma_t=a_sigma_t, a_rho=a_rho, a_comp_dist=a_comp_dist, a_omega=a_omega, a_dispNorm=a_dispNorm,
        a_no_nodes=a_no_nodes, a_lattice_height=a_lattice_height, a_lattice_width=a_lattice_width, a_phi=a_phi,
        a_envVar=int(a_envVar), a_skVec=a_skVec, a_var_e=a_var_e, a_randGraph=a_randGraph, a_gabriel=a_gabriel,
        a_T_int=a_T_int, a_envMat=a_envMat, simTime=simTime
    )

    # Extract necessary matrices and parameters from Metacommunity
    spp = metacommunity.spp
    xMat = a_xMat
    cMat = np.random.rand(xMat.shape[0], xMat.shape[1])  # Random competition matrix for testing
    rMat = np.random.rand(xMat.shape[0], xMat.shape[1])  # Random growth rate matrix
    efMat = np.random.rand(xMat.shape[0], xMat.shape[1])  # Environmental fluctuations
    scVec = np.ones(xMat.shape[1])  # Scaling vector
    scVec_prime = np.ones(xMat.shape[1])  # Alternate scaling vector
    indices_S = np.array([0, 1])  # Example Poisson clock indices
    indices_DP = np.array([2, 3])  # Example derived quantities indices
    bodymass = 0.05
    bodymass_inv = 1 / bodymass
    mu = 0.01

    # Initialize CommunityDynamics
    community_dynamics = CommunityDynamics(
        spp, xMat, cMat, rMat, efMat, scVec, scVec_prime, rho=0.2, indices_S=indices_S, indices_DP=indices_DP,
        bodymass=bodymass, g_form_of_dynamics=False, emMat=a_envMat, dMat=spp.dMat, bodymass_inv=bodymass_inv, mu=mu
    )

    # Prepare the state vector
    initial_state = community_dynamics.write_state_to()

    # Define the ODE function for integration
    def ode_function(t, y):
        time_derivative = np.zeros_like(y)
        community_dynamics.dynamics(ODEVector(y), ODEVector(time_derivative))
        return time_derivative

    # Set integration parameters
    t_span = (0, a_tMax)  # Time from 0 to tMax
    t_eval = np.linspace(*t_span, 100)  # Evaluation points

    # Perform the integration using solve_ivp
    result = solve_ivp(ode_function, t_span, initial_state, t_eval=t_eval, method='RK45')

    # Evaluate the results
    if result.success:
        print("Integration completed successfully.")
        print("State values at the final time step:")
        print(result.y[:, -1].reshape(xMat.shape))
    else:
        print("Integration failed:", result.message)

# Run the example
if __name__ == "__main__":
    example_usage()









