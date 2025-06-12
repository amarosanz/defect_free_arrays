import numpy as np

class AtomSelector:
    def __init__(self, atom_positions):
        self.atom_positions = atom_positions

    def select_nearest_atoms(self, position, num_neighbors=5):
        """
        Selecciona los átomos más cercanos a una posición dada.
        
        Parameters:
            position: Posición del átomo de referencia.
            num_neighbors: Número de átomos más cercanos a seleccionar.
        
        Returns:
            nearest_atoms: Índices de los átomos más cercanos.
        """
        distances = np.linalg.norm(self.atom_positions - position, axis=1)
        nearest_atoms = np.argsort(distances)[:num_neighbors]
        return nearest_atoms

