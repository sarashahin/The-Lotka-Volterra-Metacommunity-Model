import numpy as np
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
import logging
import scipy.io as sio

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Topography:
    def __init__(self, no_nodes, skVec, scVec,lattice_height, lattice_width, phi, envVar,
                  var_e, randGraph, gabriel, T_int, min_env=None, range_env=None, envMat=None, network_file=None, sc_file=None, env_file=None, consArea_bin=None, consArea_multiplicative=False):
        # Parameters
        self.no_nodes = no_nodes  # Number of nodes (patches) in graph
        self.lattice_height = lattice_height  # Height of lattice
        self.lattice_width = lattice_width  # Width of lattice
        self.envVar = envVar  # Number of environmental variables
        self.var_e = var_e  # Variance of the environmental distribution
        self.sigEVal = None  # Eigenvalues of spatial covariance matrix
        self.sigEVec = None  # Eigenvalues of spatial covariance matrix
        self.phi = phi  # Environmental autocorrelation length
        self.T_int = T_int  # Intercept of linear temperature gradient

        # File Paths
        self.network_file = network_file  # Path for imported network
        self.sc_file = sc_file  # Path for imported local scaling
        self.env_file = env_file  # Path for imported environment
        
        self.skVec = skVec if skVec is not None else np.array([])
        # Validate or adjust skVec to match envVar
        if len(self.skVec) != self.envVar:
            if len(self.skVec) == 0:
                self.skVec = np.ones(envVar)  # Default skVec if empty
            elif len(self.skVec) < self.envVar:
                raise ValueError(f"Provided skVec is too short (length {len(self.skVec)}); must match envVar ({self.envVar}).")
            else:
                self.skVec = self.skVec[:envVar]  # Truncate if too long


        # Switches
        self.randGraph = randGraph  # Random graph (True) or lattice (False)
        self.gabriel = gabriel  # Gabriel graph (True) or complete graph (False)

        
        # Initialize scaling vectors
        if scVec is not None:
            self.scVec = scVec
            self.scVec_prime = np.sqrt(scVec)  # Assuming scVec_prime is related in some way
        else:
            # Assign default scaling vectors if not provided
            self.scVec = np.ones(self.no_nodes)
            self.scVec_prime = np.ones(self.no_nodes)

        # Initialize range_env
        if range_env is not None:
            self.range_env = range_env
        else:
            # Default: Range of environmental variables across nodes
            self.range_env = np.linspace(0, 1, self.no_nodes)  # Uniform distribution across nodes


        # Initialize min_env
        if min_env is not None:
            self.min_env = min_env
        else:
            self.min_env = np.zeros(self.envVar)  # Default: All minimums set to 0

        # Validate dimensions
        if len(self.min_env) != self.envVar:
            raise ValueError(f"min_env length ({len(self.min_env)}) must match envVar ({self.envVar})")

        
        # Initialize environmental matrix
        self.envMat = envMat
        
        # Conservation area attributes
        self.consArea_multiplicative = consArea_multiplicative
        if consArea_bin is not None:
            if len(consArea_bin) != self.no_nodes:
                raise ValueError(f"consArea_bin length ({len(consArea_bin)}) must match no_nodes ({self.no_nodes})")
            self.consArea_bin = np.array(consArea_bin)
        else:
            self.consArea_bin = np.zeros(self.no_nodes)  # Default: No conservation areas
        
        
        # Initialize matrix objects
        self.network = None  # x, y coordinates of nodes
        self.distMat = None  # Euclidean distances between nodes
        self.sigEVec = None  # Eigenvectors of spatial covariance matrix
        self.sigEVal = None  # Eigenvalues of spatial covariance matrix
        self.adjMat = None  # Spatial adjacency matrix


        # Load or Generate the Landscape
        self.gen_landscape()


    def gen_network(self):
        """
        Generate or load a random or lattice-based spatial network.
        """
        logger.debug("Generating or importing network.")
        if self.network_file:  # Import network from the provided file
            logger.info(f"Importing network from {self.network_file}.")
            try:
                # Attempt to load binary MAT file
                if self.network_file.endswith('.mat'):
                    from scipy.io import loadmat
                    mat_data = loadmat(self.network_file)
                    self.network = mat_data.get('network')
                    if self.network is None:
                        raise ValueError("'network' key not found in MAT file.")
                else:
                    # Load from ASCII file
                    self.network = np.loadtxt(self.network_file, dtype=float, comments='#')
                
                self.no_nodes = self.network.shape[0]
                logger.debug(f"Imported network with shape: {self.network.shape}")
            except UnicodeDecodeError:
                logger.error("Failed to decode file. Attempting to read as binary.")
                self.network = np.fromfile(self.network_file, dtype=np.float32).reshape(-1, 2)
                self.no_nodes = self.network.shape[0]
            except Exception as e:
                logger.error(f"Failed to load network file: {e}")
                raise
        else:  # Generate network
            if self.randGraph:
                logger.debug("Generating random graph.")
                xcoord = np.sqrt(self.no_nodes) * np.random.rand(self.no_nodes)
                ycoord = np.sqrt(self.no_nodes) * np.random.rand(self.no_nodes)
                self.network = np.column_stack((xcoord, ycoord))
            else:  # Regular lattice as per C++ code
                logger.debug("Generating regular lattice network.")
                self.no_nodes = self.lattice_height * self.lattice_width
                xcoord = np.linspace(0, self.lattice_width - 1, self.lattice_width)
                ycoord = np.linspace(0, self.lattice_height - 1, self.lattice_height)
                xv, yv = np.meshgrid(xcoord, ycoord)
                self.network = np.column_stack((xv.ravel(), yv.ravel()))

        if self.network is None:
            raise ValueError("Network generation failed. Please verify inputs.")
        logger.debug(f"Network generated with shape: {self.network.shape}")



    def gen_dist_mat(self):
        """
        Generate the Euclidean distance matrix and covariance matrix.
        """
        logger.debug("Generating distance matrix.")
        if self.network is None:
            raise ValueError("Network has not been initialized. Please run gen_network first.")

        # Calculate Euclidean distance matrix
        self.distMat = distance_matrix(self.network, self.network)
        
        # Replace diagonal zeros with a small positive value
        np.fill_diagonal(self.distMat, 1e-6)

        # Validate `phi`
        if self.phi <= 0:
            raise ValueError("`phi` must be a positive value.")

        # Compute covariance matrix
        try:
            max_value = 700  # Prevent overflow in np.exp
            SIGMA = self.var_e * np.exp(-np.clip(self.distMat / self.phi, 0, max_value))

            # Check for inf or NaN values
            if not np.all(np.isfinite(SIGMA)):
                raise ValueError("Covariance matrix `SIGMA` contains inf or NaN values. Check `phi` and `distMat`.")

            # Eigen decomposition
            eigvals, eigvecs = eigh(SIGMA)
            eigvals = np.clip(eigvals, 0, None)  # Remove negative eigenvalues due to numerical errors
            self.sigEVec = eigvecs
            self.sigEVal = np.diag(np.sqrt(eigvals))

            logger.debug(f"Eigen decomposition completed: {eigvals}")
        except Exception as e:
            logger.error(f"Error in distance matrix or covariance calculation: {e}")
            raise




    def gen_adj_mat(self):
        """
        Generate the adjacency matrix using Gabriel or complete graph algorithm.
        """
        logger.debug("Generating adjacency matrix.")
        self.adjMat = np.zeros((self.no_nodes, self.no_nodes))

        if self.gabriel:
            for i in range(self.no_nodes):
                for j in range(i + 1, self.no_nodes):
                    mx = (self.network[i, 0] + self.network[j, 0]) / 2
                    my = (self.network[i, 1] + self.network[j, 1]) / 2
                    radius = np.sum((self.network[i] - [mx, my]) ** 2)
                    for k in range(self.no_nodes):
                        if k != i and k != j:
                            if np.sum((self.network[k] - [mx, my]) ** 2) < radius:
                                break
                    else:
                        self.adjMat[i, j] = 1
                        self.adjMat[j, i] = 1
        else:  # Complete graph
            logger.debug("Using complete graph for adjacency matrix generation.")
            self.adjMat = np.ones((self.no_nodes, self.no_nodes)) - np.eye(self.no_nodes)

        logger.debug(f"Adjacency matrix generated with shape: {self.adjMat.shape}")
        

    def gen_environment(self):
        """
        Generate or import an environmental distribution matrix using Gaussian random fields or import from a file.
        """
        logger.debug("Generating or importing environmental distribution.")
        if self.env_file:  # Import environmental data from the provided ASCII file
            logger.info(f"Importing environment matrix from {self.env_file}.")
            try:
                # Assuming the environment matrix is stored in ASCII format with rows representing different variables
                self.envMat = np.loadtxt(self.env_file)
                logger.debug(f"Imported environment matrix with shape: {self.envMat.shape}")
            except Exception as e:
                logger.error(f"Failed to load environment from ASCII file: {e}")
                raise
        else:  # Generate environment using Gaussian random fields
            if self.network is None:
                raise ValueError("Network has not been generated or loaded. Please generate network first.")

            self.envMat = np.zeros((self.envVar, self.no_nodes))
            for i in range(self.envVar):
                zVec = np.random.randn(self.no_nodes)
                eRow = (self.sigEVec @ self.sigEVal @ zVec)
                eRow = (eRow - np.mean(eRow)) / np.std(eRow)
                self.envMat[i, :] = eRow

            self.range_env = np.ptp(self.envMat, axis=1)
            self.min_env = np.min(self.envMat, axis=1)

        logger.debug("Environmental matrix generated or imported.")



    def gen_temp_grad(self):
        """
        Generate a linear temperature gradient.
        """
        logger.debug("Generating temperature gradient.")
        T_grad = 1 / np.sqrt(self.no_nodes)
        if self.network.shape[1] < 1:
            raise ValueError("Network must have at least one column for gradient calculation.")
        self.envMat = self.T_int - T_grad * self.network[:, 0]
        logger.debug(f"Generated temperature gradient with shape: {self.envMat.shape}")



    def gen_landscape(self, net_imported=None):
        """
        Generate the spatial network and domain decomposition.
        Args:
            net_imported: Imported network (numpy array) for simulation extension.
        """
        print("Generating landscape...")
        if net_imported is None or net_imported.shape[0] == 0:
            try:
                self.gen_network()  # Generate network if not imported
            except Exception as e:
                logger.error(f"Network generation failed: {e}")
                raise ValueError("Failed to generate or load network.")
        else:
            self.network = net_imported

        self.gen_dist_mat()
        self.gen_adj_mat()

        if self.envVar != 0 and not self.env_file and (self.envMat is None or self.envMat.size == 0):
            print("Generating abiotic environment...")
            self.gen_environment()

        # Initialize scaling vectors
        if self.sc_file:
            try:
                # Load scaling matrix and validate
                scaling_matrix = np.genfromtxt(self.sc_file, delimiter=' ', invalid_raise=False)
                scaling_matrix = scaling_matrix[~np.isnan(scaling_matrix).any(axis=1)]  # Remove invalid rows

                if scaling_matrix.shape[1] < 2:
                    raise ValueError("Scaling matrix must have at least two columns.")

                scVec = scaling_matrix[:, 1]
                scVec_prime = np.sqrt(scVec)

                # Handle invalid scaling vectors
                if np.all(scVec == 0):
                    logger.warning("Second column of scaling matrix is all zeros. Using first column as scVec.")
                    scVec = scaling_matrix[:, 0]
                    scVec_prime = np.sqrt(scVec)

                if np.all(scVec == 1) or np.all(scVec_prime == 1):
                    logger.warning("Scaling vectors are uniform (all ones). Verify scaling matrix file if unintended.")

                # Ensure consistency
                if len(scVec) != self.no_nodes:
                    raise ValueError(f"Scaling vector length ({len(scVec)}) does not match node count ({self.no_nodes}).")

                self.scVec = scVec
                self.scVec_prime = scVec_prime

            except Exception as e:
                logger.error(f"Error loading scaling matrix from {self.sc_file}: {e}")
                print(f"Error loading scaling matrix: {e}. Using default values.")
                self.scVec = np.ones(self.no_nodes)
                self.scVec_prime = np.ones(self.no_nodes)
        else:
            logger.info("No scaling file provided. Using default uniform scaling.")
            self.scVec = np.ones(self.no_nodes)
            self.scVec_prime = np.ones(self.no_nodes)






if __name__ == "__main__":
    topo = Topography(
        no_nodes=32,
        lattice_height=4,
        lattice_width=8,
        phi=1.0,
        envVar=2,
        var_e=1.0,
        randGraph=False,
        gabriel=True,
        scVec=np.array([0.05]),
        skVec=np.array([0.1, 0.2, 0.3]),
        T_int=25.0,
        network_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",  # Example ASCII file for network
        env_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1rMat0.mat",  # Example file for environment
        sc_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1S0.mat",  # Example file for scaling
    )
    # Generate the landscape
    topo.gen_landscape()

    # Access generated attributes
    logger.info(f"Generated network:\n{topo.network}")
    logger.info(f"Generated distance matrix:\n{topo.distMat}")
    logger.info(f"Generated adjacency matrix:\n{topo.adjMat}")
    logger.info(f"Scaling Vectors:\nscVec: {topo.scVec}\nscVec_prime: {topo.scVec_prime}")








