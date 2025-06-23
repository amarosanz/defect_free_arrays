import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lap import lapjv
import time

# Parámetros
GRID_SIZE = 4
OCCUPANCY = 0.65
TARGET_SIZE = 2
ALPHA = 1
STEPS = 20  # Número de interpolaciones

def generate_random_array(grid_size, occupancy):
    total_sites = grid_size * grid_size
    atom_count = int(total_sites * occupancy)
    indices = np.random.choice(total_sites, atom_count, replace=False)
    array = np.zeros(total_sites, dtype=bool)
    array[indices] = True
    return array.reshape((grid_size, grid_size))

def get_atom_positions(array):
    return np.argwhere(array == 1)

def get_target_positions(grid_size, target_size):
    offset = (grid_size - target_size) // 2
    # Vectorizar la generación de posiciones objetivo
    i_coords, j_coords = np.mgrid[0:target_size, 0:target_size]
    positions = np.column_stack([
        (i_coords + offset).ravel(), 
        (j_coords + offset).ravel()
    ])
    return positions

def compute_cost_matrix_vectorized(atoms, targets, alpha=1.0):
    """
    Versión completamente vectorizada del cálculo de matriz de costos
    """
    # Expandir dimensiones para broadcasting
    # atoms: (N_atoms, 2) -> (1, N_atoms, 2)
    # targets: (N_targets, 2) -> (N_targets, 1, 2)
    atoms_expanded = atoms[np.newaxis, :, :]      # (1, N_atoms, 2)
    targets_expanded = targets[:, np.newaxis, :]  # (N_targets, 1, 2)
    
    # Calcular diferencias: (N_targets, N_atoms, 2)
    diff = targets_expanded - atoms_expanded
    
    # Calcular distancias: (N_targets, N_atoms)
    distances = np.linalg.norm(diff, axis=2)
    
    # Aplicar potencia alpha
    cost_matrix = distances ** alpha
    print(f"Cost matrix shape: {cost_matrix.shape}")
    print(f"Cost matrix: {cost_matrix}")
    
    return cost_matrix

def find_discarded_atoms_vectorized(all_atoms, selected_atoms):
    """
    Versión vectorizada para encontrar átomos descartados
    """
    if len(selected_atoms) == 0:
        return all_atoms
    
    # Usar broadcasting para comparar todos los átomos con los seleccionados
    # all_atoms: (N_all, 2) -> (N_all, 1, 2)
    # selected_atoms: (N_sel, 2) -> (1, N_sel, 2)
    all_expanded = all_atoms[:, np.newaxis, :]
    # print(f"all expanded array: {all_expanded}")

    selected_expanded = selected_atoms[np.newaxis, :, :]
    # print(f"selected expanded array: {selected_expanded}")
    # Verificar igualdad elemento por elemento y luego por filas completas
    equal_elements = (all_expanded == selected_expanded)  # (N_all, N_sel, 2)
    equal_rows = np.all(equal_elements, axis=2)           # (N_all, N_sel)
    is_selected = np.any(equal_rows, axis=1)              # (N_all,)
    
    # Retornar los que NO están seleccionados
    return all_atoms[~is_selected]

def select_closest_atoms_vectorized(atoms, center, n_targets):
    """
    Versión vectorizada para seleccionar átomos más cercanos al centro
    """
    # Calcular todas las distancias de una vez
    distances = np.linalg.norm(atoms - center, axis=1)
    print(f"Center: {center}")
    # Obtener índices ordenados y seleccionar los primeros n_targets
    sorted_indices = np.argpartition(distances, min(n_targets-1, len(distances)-1))[:n_targets]
    return atoms[sorted_indices], sorted_indices

def plot_rearrangement(all_atoms, selected_atoms, final_targets, assignment, grid_size, assign_time):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Inicial
    ax = axes[0]
    ax.set_title("Array inicial")
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True)
    # Vectorizar el plotting
    if len(all_atoms) > 0:
        ax.scatter(all_atoms[:, 1], grid_size - 1 - all_atoms[:, 0], c='blue', s=20)

    # Asignaciones
    ax = axes[1]
    ax.set_title(f"Asignaciones (Hungarian)\nTiempo: {assign_time:.4f} s")
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True)
    
    
# Encontrar átomos descartados de forma vectorizada
    discarded = find_discarded_atoms_vectorized(all_atoms, selected_atoms)
    
    
    # Primero las cruces rojas (debajo)
    if len(final_targets) > 0:
        ax.scatter(final_targets[:, 1], grid_size - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)

    # Átomos descartados
    if len(discarded) > 0:
        ax.scatter(discarded[:, 1], grid_size - 1 - discarded[:, 0], c='gray', s=20, zorder=2)

    # Clasificar átomos que no se mueven
    unmoved_mask = np.all(selected_atoms == final_targets[assignment], axis=1)
    moved_mask = ~unmoved_mask
    selected_unmoved = selected_atoms[unmoved_mask]
    selected_moved = selected_atoms[moved_mask]

    # Átomos no movidos (verdes)
    if len(selected_unmoved) > 0:
        ax.scatter(selected_unmoved[:, 1], grid_size - 1 - selected_unmoved[:, 0], c='green', s=20, zorder=3)
    # Átomos movidos (azules)
    if len(selected_moved) > 0:
        ax.scatter(selected_moved[:, 1], grid_size - 1 - selected_moved[:, 0], c='blue', s=20, zorder=3)
    # Dibujar flechas (esto es difícil de vectorizar completamente)
    for i, j in enumerate(assignment):
        if i < len(final_targets) and j < len(selected_atoms):
            start = selected_atoms[j]
            end = final_targets[i]
            ax.annotate("",
                        xy=(end[1], grid_size - 1 - end[0]),
                        xytext=(start[1], grid_size - 1 - start[0]),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    # Final
    ax = axes[2]
    ax.set_title("Array final sin defectos")
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True)
    if len(final_targets) > 0:
        ax.scatter(final_targets[:, 1], grid_size - 1 - final_targets[:, 0], c='red', s=20)

    plt.tight_layout()
    plt.show()

def plot_trajectory_slider(selected_atoms, final_targets, grid_size, steps, discarded_atoms, assign_time):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(f"Interpolación del movimiento atómico\nTiempo de asignación: {assign_time:.4f} s")

    # Puntos móviles (interpolados)
    scatter = ax.scatter([], [], c='blue', s=20, zorder=2)
    targets_plot = ax.scatter(final_targets[:, 1], grid_size - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)
    discarded_plot = ax.scatter([], [], c='gray', s=20, zorder=2)

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Paso', 0, steps, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
        # Interpolación vectorizada
        t = step / steps
        interp_pos = (1 - t) * selected_atoms + t * final_targets
        
        # Actualizar posiciones
        scatter.set_offsets(np.column_stack([interp_pos[:, 1], grid_size - 1 - interp_pos[:, 0]]))

        if step == 0 and len(discarded_atoms) > 0:
            discarded_plot.set_offsets(np.column_stack([discarded_atoms[:, 1], grid_size - 1 - discarded_atoms[:, 0]]))
        else:
            discarded_plot.set_offsets(np.empty((0, 2)))

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

def main():
    print("Generando array inicial...")
    array = generate_random_array(GRID_SIZE, OCCUPANCY)
    atoms = get_atom_positions(array)
    print(f"Posiciones de los átomos: {atoms}")
    targets = get_target_positions(GRID_SIZE, TARGET_SIZE)
    print(f"Posiciones de los objetivos: {targets}")
    
    print(f"Átomos encontrados: {len(atoms)}")
    print(f"Posiciones objetivo: {len(targets)}")

    if len(atoms) < len(targets):
        print("No hay suficientes átomos para llenar el array sin defectos.")
        return

    # Selección vectorizada de átomos más cercanos al centro
    center = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
    selected_atoms, selected_indices = select_closest_atoms_vectorized(atoms, center, len(targets))
    
    print("Calculando matriz de costos...")
    start_cost_time = time.perf_counter()
    cost_matrix = compute_cost_matrix_vectorized(selected_atoms, targets, alpha=ALPHA)
    cost_time = time.perf_counter() - start_cost_time
    print(f"Tiempo de cálculo de matriz de costos: {cost_time:.6f} segundos")

    print("Resolviendo asignación húngara...")
    start_time = time.perf_counter()
    row_ind, col_ind, total_cost = lapjv(cost_matrix)
    print(f"Row indices: {row_ind}")
    print(f"Column indices: {col_ind}")
    print(f"Total cost: {total_cost}")      
    end_time = time.perf_counter()
    assign_time = end_time - start_time

    print(f"Tiempo de asignación húngara: {assign_time:.6f} segundos")
    print(f"Tiempo total de optimización: {cost_time + assign_time:.6f} segundos")

    assigned_atoms = selected_atoms[col_ind]
    discarded_atoms = find_discarded_atoms_vectorized(atoms, assigned_atoms)

    print(f"Átomos asignados: {len(assigned_atoms)}")
    print(f"Átomos descartados: {len(discarded_atoms)}")

    plot_rearrangement(atoms, assigned_atoms, targets, np.arange(len(targets)), GRID_SIZE, assign_time)
    plot_trajectory_slider(assigned_atoms, targets, GRID_SIZE, steps=STEPS,
                           discarded_atoms=discarded_atoms, assign_time=assign_time)

if __name__ == "__main__":
    main()