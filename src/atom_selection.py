import numpy as np

# This module provides a class for selecting the atoms to be moved in the assignment procedure.

class AtomSelector:
    def __init__(self, 
                 atom_positions : np.ndarray,  # Array of atom positions
                 target_positions: np.ndarray  # Array of target positions 
                 ):
        # instance variables
        self.atom_positions = atom_positions
        self.target_positions = target_positions

    # select closest atoms 
    def select_closest_atoms(self, center: np.ndarray, n_targets: int):
        """
        Select the closest atoms to a given center point.

        Args:
            center (np.ndarray): The center point from which to measure distances.
            n_targets (int): The number of closest atoms to select.

        Returns:
            np.ndarray: The selected closest atoms.
            np.ndarray: Indices of the selected atoms in the original atom positions.
        """
        distances = np.linalg.norm(self.atom_positions - center, axis=1)
        sorted_indices = np.argpartition(distances, min(n_targets-1, len(distances)-1))[:n_targets]
        return self.atom_positions[sorted_indices], sorted_indices
    # find discarded atoms 