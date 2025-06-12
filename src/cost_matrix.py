import numpy as np

class CostMatrix:
    def __init__(self, positions1, positions2):
        self.positions1 = positions1
        self.positions2 = positions2
        self.matrix = self.calculate_cost_matrix()

    def calculate_cost_matrix(self):
        """
        Calcula una matriz de costos basada en la distancia euclidiana entre las posiciones de dos conjuntos de átomos.
        
        Returns:
            cost_matrix: Matriz de costos entre las posiciones.
        """
        cost_matrix = np.linalg.norm(self.positions1[:, np.newaxis] - self.positions2, axis=2)
        return cost_matrix

