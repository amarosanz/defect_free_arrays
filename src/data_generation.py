import numpy as np

# This module provides a data generator for creating random square grids of atoms. It may be desirable 
# to extend it to more complex atomic arrangements in the future.

class DataGenerator:
    def __init__(self, 
                 grid_size : int = 40, #size of the square grid
                 target_type : str = "square", #type of target area, currently only "square" is supported
                 target_size : int = 25, #size of the target square grid
                 occupancy = 0.65 #Ratio of occupied sites
                 
                ):
        #instance variables
        self.grid_size = grid_size
        self.target_type = target_type
        self.target_size = target_size
        self.occupancy = occupancy 
        self.total_sites = grid_size * grid_size
        self.atom_count = int(self.total_sites * occupancy)
        if target_type == "square":
            self.target_sites = target_size * target_size
            self.target_center = np.array([grid_size // 2, grid_size // 2]) #center of the target area

        #some constraints
        if not (0 < target_size <= grid_size):
            raise ValueError("Target size must be between 0 and grid size.") 
        
        if not (0 < occupancy <= 1):
            raise ValueError("Occupancy must be between 0 and 1.")

    def generate_random_array(self):
        """
        Generate random boolean array.

        Returns:
            array: Random boolean array of shape (grid_size, grid_size).
        """
        indices = np.random.choice(self.total_sites, self.atom_count, replace=False) #indices of occupied sites
        array = np.zeros(self.total_sites, dtype=bool)
        array[indices] = True #shape of the array is (total_sites,)
        array = array.reshape((self.grid_size, self.grid_size)) #reshape to (grid_size, grid_size
        return array
    
    def get_atom_positions(self):
        """
        Get positions of atoms in the grid.
        
        Returns:
            positions: Array of atom positions.
        """
        array = self.generate_random_array()
        positions = np.argwhere(array == 1)
        return positions  
    

    def get_target_positions(self):
        """
        Get positions of target atoms in the grid given the initial grid and target sizes
        
        Returns:
            positions: Array of target positions.
        """
        offset = (self.grid_size - self.target_size) // 2
        i_coords, j_coords = np.mgrid[0:self.target_size, 0:self.target_size]
        positions = np.column_stack([(i_coords + offset).ravel(), (j_coords + offset).ravel()])
        return positions


    def get_target_positions_from_matrix(self, binary_matrix):
        """
        Get positions of target (bright) pixels from the corresponding binary matrix.
        Assumes (0,0) is the bottom-left corner of the image.

        Args:
        binary_matrix: 2D numpy array with 0s and 1s

        Returns:
        positions: Array of (i, j) positions where value == 1, with origin at bottom-left
        """
        # Get positions where matrix value is 1
        positions = np.argwhere(binary_matrix == 1)
        
        # Convert to Cartesian-style coordinates (origin at bottom-left)
        h = binary_matrix.shape[0]
        cartesian_positions = np.column_stack((h - 1 - positions[:, 0], positions[:, 1]))
        
        return cartesian_positions
    
    