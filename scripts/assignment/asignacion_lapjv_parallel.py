import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lap import lapjv
import time
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor

#To be structured in a modular fashion

# Parámetros
GRID_SIZE = 40
OCCUPANCY = 0.65
TARGET_SIZE = 25
ALPHA = 2
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
    i_coords, j_coords = np.mgrid[0:target_size, 0:target_size]
    positions = np.column_stack([(i_coords + offset).ravel(), (j_coords + offset).ravel()])
    return positions

def compute_cost_matrix_scipy(atoms, targets, alpha=1.0):
    if alpha == 1.0:
        return cdist(targets, atoms, metric='euclidean')
    elif alpha == 2.0:
        return cdist(targets, atoms, metric='sqeuclidean')
    else:
        distances = cdist(targets, atoms, metric='euclidean')
        return distances ** alpha

def find_discarded_atoms_vectorized(all_atoms, selected_atoms):
    if len(selected_atoms) == 0:
        return all_atoms
    all_expanded = all_atoms[:, np.newaxis, :]
    selected_expanded = selected_atoms[np.newaxis, :, :]
    equal_elements = (all_expanded == selected_expanded)
    equal_rows = np.all(equal_elements, axis=2)
    is_selected = np.any(equal_rows, axis=1)
    return all_atoms[~is_selected]

def select_closest_atoms_vectorized(atoms, center, n_targets):
    distances = np.linalg.norm(atoms - center, axis=1)
    sorted_indices = np.argpartition(distances, min(n_targets-1, len(distances)-1))[:n_targets]
    return atoms[sorted_indices], sorted_indices

def assign_subregion(atoms, targets, alpha):
    cost_matrix = compute_cost_matrix_scipy(atoms, targets, alpha)
    _, col_ind, _ = lapjv(cost_matrix)
    assigned_atoms = atoms[col_ind]
    return assigned_atoms

def plot_rearrangement(all_atoms, selected_atoms, final_targets, assignment, grid_size, assign_time):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.set_title("Array inicial")
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True)
    if len(all_atoms) > 0:
        ax.scatter(all_atoms[:, 1], grid_size - 1 - all_atoms[:, 0], c='blue', s=20)

    ax = axes[1]
    ax.set_title(f"Asignaciones (Paralelo)\nTiempo: {assign_time:.4f} s")
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.grid(True)

    discarded = find_discarded_atoms_vectorized(all_atoms, selected_atoms)

    if len(final_targets) > 0:
        ax.scatter(final_targets[:, 1], grid_size - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)

    if len(discarded) > 0:
        ax.scatter(discarded[:, 1], grid_size - 1 - discarded[:, 0], c='gray', s=20, zorder=2)

    unmoved_mask = np.all(selected_atoms == final_targets[assignment], axis=1)
    moved_mask = ~unmoved_mask
    selected_unmoved = selected_atoms[unmoved_mask]
    selected_moved = selected_atoms[moved_mask]

    if len(selected_unmoved) > 0:
        ax.scatter(selected_unmoved[:, 1], grid_size - 1 - selected_unmoved[:, 0], c='green', s=20, zorder=3)
    if len(selected_moved) > 0:
        ax.scatter(selected_moved[:, 1], grid_size - 1 - selected_moved[:, 0], c='blue', s=20, zorder=3)
    for i, j in enumerate(assignment):
        if i < len(final_targets) and j < len(selected_atoms):
            start = selected_atoms[j]
            end = final_targets[i]
            ax.annotate("", xy=(end[1], grid_size - 1 - end[0]), xytext=(start[1], grid_size - 1 - start[0]),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

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

    scatter = ax.scatter([], [], c='blue', s=20, zorder=2)
    targets_plot = ax.scatter(final_targets[:, 1], grid_size - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)
    discarded_plot = ax.scatter([], [], c='gray', s=20, zorder=2)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Paso', 0, steps, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
        t = step / steps
        interp_pos = (1 - t) * selected_atoms + t * final_targets
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
    targets = get_target_positions(GRID_SIZE, TARGET_SIZE)

    print(f"Átomos encontrados: {len(atoms)}")
    print(f"Posiciones objetivo: {len(targets)}")

    if len(atoms) < len(targets):
        print("No hay suficientes átomos para llenar el array sin defectos.")
        return

    mid = GRID_SIZE // 2
    quadrants = []
    for i in range(2):
        for j in range(2):
            target_mask = (
                (targets[:, 0] >= i * mid) & (targets[:, 0] < (i + 1) * mid) &
                (targets[:, 1] >= j * mid) & (targets[:, 1] < (j + 1) * mid)
            )
            targets_q = targets[target_mask]
            center_q = np.array([(2 * i + 1) * mid // 2, (2 * j + 1) * mid // 2])

            atom_mask = (
                (atoms[:, 0] >= i * mid) & (atoms[:, 0] < (i + 1) * mid) &
                (atoms[:, 1] >= j * mid) & (atoms[:, 1] < (j + 1) * mid)
            )
            atoms_q = atoms[atom_mask]

            if len(atoms_q) >= len(targets_q):
                selected_q, _ = select_closest_atoms_vectorized(atoms_q, center_q, len(targets_q))
                quadrants.append((selected_q, targets_q))
            else:
                print(f"Cuadrante ({i},{j}) no tiene suficientes átomos. Tiene {len(atoms_q)} y necesita {len(targets_q)}.")
                return

    print("Ejecutando asignaciones en paralelo...")
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda args: assign_subregion(*args, ALPHA), quadrants))
    end_time = time.perf_counter()
    assign_time = end_time - start_time
    print(f"Tiempo total de asignación paralela: {assign_time:.6f} segundos")

    assigned_atoms_blocks = []
    targets_blocks = []
    for (atoms_q, targets_q), assigned_q in zip(quadrants, results):
        assigned_atoms_blocks.append(assigned_q)
        targets_blocks.append(targets_q)

    assigned_atoms = np.vstack(assigned_atoms_blocks)
    targets_ordered = np.vstack(targets_blocks)
    assignment = np.arange(len(targets_ordered))

    discarded_atoms = find_discarded_atoms_vectorized(atoms, assigned_atoms)

    plot_rearrangement(atoms, assigned_atoms, targets_ordered, assignment, GRID_SIZE, assign_time)
    plot_trajectory_slider(assigned_atoms, targets_ordered, GRID_SIZE, steps=STEPS,
                           discarded_atoms=discarded_atoms, assign_time=assign_time)

if __name__ == "__main__":
    main()
