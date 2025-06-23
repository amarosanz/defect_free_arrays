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


    def find_discarded_atoms(self, selected_atoms: np.ndarray):
        """
        Find atoms that are not in the selected set.

        Args:
            selected_atoms (np.ndarray): The array of selected atom positions.

        Returns:
            np.ndarray: The atoms that are not selected.
        """
        if len(selected_atoms) == 0:
            return self.atom_positions
        
        # We add a new axis to both the atomic positions and the selected atoms for array broadcasting:
        # all_atoms: (n_all, 2) -> (n_all, 1, 2)
        # selected_atoms: (n_selected, 2) -> (1, n_selected, 2)

        all_expanded = self.atom_positions[:, np.newaxis, :]
        selected_expanded = selected_atoms[np.newaxis, :, :]
        
        # When the 2 arrays are compared, Numpy will broadcast them to a common shape (n_all, n_selected, 2)
        # We build a boolean array where each element tells whether the 
        # corresponding elements in the two arrays are equal, i.e., which atoms are selected.

        equal_elements = (all_expanded == selected_expanded) #shape (n_all, n_selected, 2)

        # We take the equal_elements array and see if all elements in each row are True
        equal_rows = np.all(equal_elements, axis=2) #shape (n_all, n_selected)

        # Finally, if any of the elements in the rows of equal_rows is True, it means that the corresponding atom is selected
        is_selected = np.any(equal_rows, axis=1) #shape (n_all,)

        # We return the atoms that are NOT selected        
        return self.atom_positions[~is_selected]