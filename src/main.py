from src.data_generation import DataGenerator
from src.cost_matrix import CostMatrix
from src.atom_selection import AtomSelector
from src.plot_utils import Plotter

def main():
    # Crear instancias de las clases
    data_gen = DataGenerator()
    atom_positions1 = data_gen.get_target_positions(10)
    atom_positions2 = data_gen.get_target_positions(10)
    
    # Calcular la matriz de coste
    cost_matrix = CostMatrix(atom_positions1, atom_positions2)

    # Seleccionar los 3 átomos más cercanos a un átomo específico
    atom_selector = AtomSelector(atom_positions2)
    nearest_atoms = atom_selector.select_nearest_atoms(atom_positions1[0], num_neighbors=3)
    print(f"Átomos más cercanos: {nearest_atoms}")
    
    # Graficar las posiciones de los átomos
    plotter = Plotter()
    plotter.plot_atom_positions(atom_positions1)
    
    # Mostrar slider de interpolación (si aplica)
    plotter.plot_interpolation_slider(cost_matrix.matrix)

if __name__ == "__main__":
    main()

