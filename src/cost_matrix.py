import numpy as np
from scipy.spatial.distance import cdist
from lap import lapjv


# This module provides two classes: CostMatrix and Solver: 
#   - CostMatrix is an object representing 
#   - Solver is a class for solving the assignment problem given a cost matrix. For now
#     it only implements the LapJV algorithm (a variant of the well-known Hungarian method).

class CostMatrix:
    def __init__(self, 
                 atom_positions: np.ndarray, # Array of atom positions
                 target_positions: np.ndarray, # Array of target positions
                 alpha: float = 2.0): # Power to which the distances are raised in the cost matrix calculation (if Hungarian method is used)

        #instance variables
        self.atom_positions = atom_positions
        self.target_positions = target_positions
        self.alpha = alpha

    def compute_cost_matrix(self):
        """
        Compute the cost matrix based on the positions of atoms and targets.
        The cost matrix is computed as the distance between each target and atom raised to the power of alpha.
        If alpha is 1.0, it uses Euclidean distance; if alpha is 2.0, it uses squared Euclidean distance.
        For intermediate values of alpha, it raises the Euclidean distance to the power of alpha.
        Returns:
            np.ndarray: The cost matrix of shape (N_targets, N_atoms).      
        """
        if self.alpha == 1.0:
            return cdist(self.atom_positions, self.target_positions, metric='euclidean')
        elif self.alpha == 2.0:
            return cdist(self.atom_positions, self.target_positions, metric='sqeuclidean')
        else:
            distances = cdist(self.target_positions, self.atom_positions, metric='euclidean')
            cost_matrix = distances ** self.alpha
            return cost_matrix

class Solver:
    def __init__(self, 
                 cost_matrix: np.ndarray,
                 method: str = 'lapjv'):
        """
        Initialize the solver with a cost matrix.
        
        Args:
            cost_matrix (np.ndarray): The cost matrix to be solved.
        """
        self.cost_matrix = cost_matrix

    def lapjv(self):
        """
        Solve the assignment problem using the LAPJV algorithm.
        
        Returns:
            tuple: A tuple containing the indices of the optimal assignment and the total cost.
        """
        total_cost, col_ind, row_ind = lapjv(self.cost_matrix)
        print(f"Row indices: {row_ind}")
        print(f"Column indices: {col_ind}")
        print(f"Total cost: {total_cost}")          
        return total_cost, col_ind, row_ind

