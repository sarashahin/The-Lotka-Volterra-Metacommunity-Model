import random
from numpy.random import MT19937, SeedSequence, RandomState

class LVMCM_rng:
    # Use Python's random generator or NumPy's MT19937 for reproducibility
    def __init__(self, seed=None):
        # If a seed is provided, use it; otherwise, use a random seed
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.rng = RandomState(MT19937(SeedSequence(self.seed)))

    def set_seed(self, new_seed):
        # Update the seed and create a new RandomState with that seed
        self.seed = new_seed
        self.rng = RandomState(MT19937(SeedSequence(self.seed)))
        random.seed(self.seed)  # Set Python's random generator seed as well for consistency

# Example usage:
rng_instance = LVMCM_rng(seed=1)
rng_instance.set_seed(42)  # Dynamically change the seed
