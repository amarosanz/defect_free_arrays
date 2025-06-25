
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from tests import BenchmarkTester, BenchmarkPlotter

if __name__ == "__main__":
    tester = BenchmarkTester(grid_size=120, alpha=2)
    plotter = BenchmarkPlotter()

    # Benchmark 1: assignment time heatmap as a function of reservoir size and target size
    sizes, heatmap = tester.benchmark_time_heatmap_sizes(min_size=2, max_size=10, runs=10)
    plotter.time_heatmap_size(sizes, heatmap, save_path="assets/benchmark_results/size_heatmap.pdf")

    # Benchmark 2: assignment time heatmap as a function of target size and occupancy
    reservoir_size = 20
    target_range = list(range(2, 21))
    occupancy_range = np.round(np.linspace(0.1, 1.0, 10), 2)
    target_sizes, occs, occ_heatmap = tester.benchmark_time_heatmap_occupancy_targetsize(reservoir_size, target_range, occupancy_range)
    plotter.time_heatmap_occupancy(target_sizes, occs, occ_heatmap, save_path="assets/benchmark_results/occupancy_map.pdf")

    # Benchmark 3: assignment time vs. target size (with error bars)
    reservoir_size = 20
    target_range = list(range(2, 21))
    occupancy = 0.65
    target_sizes, means, stds = tester.benchmark_assignment_time_vs_target_atoms(
        reservoir_size=reservoir_size,
        target_range=target_range,
        occupancy=occupancy,
        runs=100
    )
    plotter.time_vs_size(
        target_sizes, means, stds,
        save_path="assets/benchmark_results/target_vs_time.pdf"
    ) 






# OLD 

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to the system path
# from src.data_generation import DataGenerator
# from src.cost_matrix import CostMatrix, Solver
# from src.atom_selection import AtomSelector
# from src.plot_utils import Plotter
# import time


# def main():
#     # Crear instancias de las clases
#     data_gen = DataGenerator(grid_size = 100, target_size= 2)
#     atom_positions1 = data_gen.get_atom_positions()
#     print(f"Posiciones de los átomos: {atom_positions1}")
#     atom_positions2 = data_gen.get_target_positions()
#     print(f"Posiciones de los objetivos: {atom_positions2}")
#     atom_selector = AtomSelector(atom_positions1, atom_positions2)
#     center = data_gen.target_center
#     print(f"Centro del área objetivo: {center}")
#     ntargets = data_gen.target_sites
#     selected_atoms, _ = atom_selector.select_closest_atoms(center, ntargets)
#     atom_selector.find_discarded_atoms(selected_atoms)
#     cost_matrix_object = CostMatrix(selected_atoms, atom_positions2, alpha = 2)
#     cost_matrix = cost_matrix_object.compute_cost_matrix()

#     # print(f"Matriz de coste: {cost_matrix}")
#     print(f"Forma de la matriz de coste: {cost_matrix.shape}")
#     solv = Solver(cost_matrix)
#     initial_time = time.perf_counter()
#     total_cost, col_ind, row_ind = solv.lapjv()
#     final_time = time.perf_counter()
#     assignment_time = (final_time - initial_time)*1000

#     print(f"Assignment time: {assignment_time:.1f} ms")

    

# if __name__ == "__main__":
#     main()