import numpy as np

class DataGenerator:
    def __init__(self, 
                 grid_size : int = 40, #size of the square grid
                 target_size : int = 25, #size of the target square grid
                 occupancy = 0.65 #Ratio of occupied sites
                 
                ):
        #instance variables
        self.grid_size = grid_size
        self.target_size = target_size
        self.occupancy = occupancy 
        self.total_sites = grid_size * grid_size
        self.atom_count = int(self.total_sites * occupancy)
        self.target_sites = target_size * target_size

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
        i_coords, j_coords = np.mgrid[0:self.atom_count, 0:self.atom_count]
        positions = np.column_stack([(i_coords + offset).ravel(), (j_coords + offset).ravel()])
        return positions
