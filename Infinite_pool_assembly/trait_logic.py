############################################
# trait_logic.py
############################################
"""
Encapsulates the mapping between abstract species traits,
environmental location, and demographic parameters (r, C).
"""
import numpy as np
from config import INTERACTION_STRENGTH, CONNECTANCE, NUM_PATCHES_X, NUM_PATCHES_Y

TWO_TYPES = False

class TraitManager:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.environment = None
        self._init_environment()

    def _init_environment(self):
        """
        Creates a static environmental map for the grid.
        Scenario: Split world.
        - Patches 0 to N/2 -> Environment Type 0 (e.g., Forest)
        - Patches N/2 to N -> Environment Type 1 (e.g., Grassland)
        """
        n_patches = NUM_PATCHES_X * NUM_PATCHES_Y
        self.env_map = np.zeros(n_patches, dtype=int)
        
        # Simple vertical split
        # We can map 2D coordinates if needed, but linear split is fine for testing
        if TWO_TYPES:
            split_idx = n_patches // 4
            self.env_map[split_idx:] = 1

    def generate_traits(self, n_new):
        """
        Sample abstract traits for n_new new species.
        Current Rule: Integer 0 or 1.
        """
        if TWO_TYPES:
            return self.rng.integers(0, 2, size=n_new)
        else:
            return np.zeros(n_new, dtype=int)

    def get_growth_rates(self, traits):
        """
        Calculate spatially dependent intrinsic growth rates (r).
        Returns shape (n_new, n_patches).
        
        Rules:
          - Species Trait 0 prefers Env 0 (r=1.0) and dislikes Env 1 (r=0.8)
          - Species Trait 1 prefers Env 1 (r=1.0) and dislikes Env 0 (r=0.8)
        """
        n_new = len(traits)
        n_patches = len(self.env_map)
        
        # Initialize output field (S, P)
        r_field = np.zeros((n_new, n_patches), dtype=float)
        
        # Broadcast environment to shape (1, P)
        env_broadcast = self.env_map[np.newaxis, :]
        
        # Broadcast traits to shape (S, 1)
        traits_broadcast = traits[:, np.newaxis]
        
        # Logic: Match = 1.0, Mismatch = 0.8
        # (You can add random noise here too if desired)
        r_field = np.where(traits_broadcast == env_broadcast, 1.0, 0.8)
        
        return r_field


    def get_interaction_strengths(self, target_traits, source_traits):
        """
        Calculate interaction coefficients C[i, j] (effect OF source ON target).

        Args:
            target_traits: Traits of the species being affected.
            source_traits: Traits of the species causing the effect.

        Returns:
            Array of interaction strengths.
        """
        n = len(target_traits)
        

        # Current rule: interactions don't depend on traits. We use
        # CONNECTANCE and INTERACTION_STRENGTH from config.

        mask = self.rng.random(n) < CONNECTANCE
        interactions = np.zeros(n)
        interactions[mask] = INTERACTION_STRENGTH         # constant value

        return interactions
