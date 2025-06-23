import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation import DataGenerator
from src.cost_matrix import CostMatrix, Solver
from src.atom_selection import AtomSelector
from src.plot_utils import Plotter

def main():
    # Crear instancias de las clases
    data_gen = DataGenerator(grid_size = 5, target_size= 3)
    atom_positions1 = data_gen.get_atom_positions()
    print(f"Posiciones de los átomos: {atom_positions1}")
    atom_positions2 = data_gen.get_target_positions()
    print(f"Posiciones de los objetivos: {atom_positions2}")
    atom_selector = AtomSelector(atom_positions1, atom_positions2)
    center = data_gen.target_center
    print(f"Centro del área objetivo: {center}")
    ntargets = data_gen.target_sites
    selected_atoms, _ = atom_selector.select_closest_atoms(center, ntargets)
    atom_selector.find_discarded_atoms(selected_atoms)
    cost_matrix_object = CostMatrix(selected_atoms, atom_positions2, alpha = 2)
    cost_matrix = cost_matrix_object.compute_cost_matrix()

    # print(f"Matriz de coste: {cost_matrix}")
    print(f"Forma de la matriz de coste: {cost_matrix.shape}")
    solv = Solver(cost_matrix)
    total_cost, col_ind, row_ind = solv.lapjv()

    
    # # Calcular la matriz de coste
    # cost_matrix = CostMatrix(atom_positions1, atom_positions2)

    # # Seleccionar los 3 átomos más cercanos a un átomo específico
    # atom_selector = AtomSelector(atom_positions2)
    # nearest_atoms = atom_selector.select_nearest_atoms(atom_positions1[0], num_neighbors=3)
    # print(f"Átomos más cercanos: {nearest_atoms}")
    
    # # Graficar las posiciones de los átomos
    # plotter = Plotter()
    # plotter.plot_atom_positions(atom_positions1)
    
    # # Mostrar slider de interpolación (si aplica)
    # plotter.plot_interpolation_slider(cost_matrix.matrix)

if __name__ == "__main__":
    main()

