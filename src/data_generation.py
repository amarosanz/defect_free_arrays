import numpy as np

class DataGenerator:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    def generate_random_array(self, size):
        """
        Genera un array de números aleatorios de tamaño 'size'.
        """
        return np.random.random(size)

    def get_target_positions(self, num_atoms):
        """
        Genera posiciones aleatorias de átomos en una cuadrícula.
        
        Parameters:
            num_atoms: Número de átomos a colocar.
        
        Returns:
            positions: Lista de posiciones de los átomos.
        """
        positions = np.random.randint(0, self.grid_size, size=(num_atoms, 2))
        return positions

