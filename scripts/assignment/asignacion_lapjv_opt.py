import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lap import lapjv
import time
from scipy.spatial.distance import cdist

GRID_SIZE = 40
OCCUPANCY = 0.65
TARGET_SIZE = 25
ALPHA = 2
STEPS = 20
SEED = None

def generate_random_array(grid_size: int, occupancy: float, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_sites = grid_size * grid_size
    atom_count = int(total_sites * occupancy)
    indices = rng.choice(total_sites, atom_count, replace=False)
    array = np.zeros(total_sites, dtype=bool)
    array[indices] = True
    return array.reshape((grid_size, grid_size))

def get_atom_positions(array: np.ndarray) -> np.ndarray:
    return np.argwhere(array)

def get_target_positions(grid_size: int, target_size: int) -> np.ndarray:
    offset = (grid_size - target_size) // 2
    i, j = np.mgrid[0:target_size, 0:target_size]
    return np.column_stack([(i + offset).ravel(), (j + offset).ravel()])

def compute_cost_matrix(atoms: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if alpha == 1.0:
        return cdist(targets, atoms, metric='euclidean')
    elif alpha == 2.0:
        return cdist(targets, atoms, metric='sqeuclidean')
    dist = cdist(targets, atoms, metric='euclidean')
    return dist ** alpha

def simple_global_assignment(atoms: np.ndarray, targets: np.ndarray, alpha: float = 2.0):
    """
    LA SOLUCIÓN REAL: Una sola asignación global sin divisiones
    """
    # Tomar solo los mejores átomos (los más cercanos al centro del target)
    target_center = np.mean(targets, axis=0)
    distances = np.linalg.norm(atoms - target_center, axis=1)
    best_atoms_idx = np.argpartition(distances, len(targets))[:len(targets)]
    selected_atoms = atoms[best_atoms_idx]
    
    # Una sola asignación óptima global
    cost = compute_cost_matrix(selected_atoms, targets, alpha)
    t0 = time.perf_counter()
    _, cols, _ = lapjv(cost)
    t_assign = time.perf_counter() - t0
    
    return selected_atoms[cols], targets, np.arange(len(targets)), t_assign

def find_discarded_atoms(all_atoms: np.ndarray, selected_atoms: np.ndarray):
    if len(selected_atoms) == 0:
        return all_atoms
    a = all_atoms[:, None, :] == selected_atoms[None, :, :]
    is_sel = np.all(a, axis=2).any(axis=1)
    return all_atoms[~is_sel]

def _plot_rearrangement(all_atoms, selected_atoms, final_targets, assignment, gsize, t):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    
    ax = axs[0]
    ax.set_title('Array inicial')
    ax.set_xlim(-1, gsize)
    ax.set_ylim(-1, gsize)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.scatter(all_atoms[:, 1], gsize - 1 - all_atoms[:, 0], c='blue', s=20)
    
    ax = axs[1]
    ax.set_title(f'Asignación Global (SIN CRUCES)\nTiempo: {t:.4f} s')
    ax.set_xlim(-1, gsize)
    ax.set_ylim(-1, gsize)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.scatter(final_targets[:, 1], gsize - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)
    
    disc = find_discarded_atoms(all_atoms, selected_atoms)
    if disc.size:
        ax.scatter(disc[:, 1], gsize - 1 - disc[:, 0], c='gray', s=20, zorder=2)
    ax.scatter(selected_atoms[:, 1], gsize - 1 - selected_atoms[:, 0], c='blue', s=20, zorder=2)
    
    for i, j in enumerate(assignment):
        ax.annotate('', xy=(final_targets[i, 1], gsize - 1 - final_targets[i, 0]),
                    xytext=(selected_atoms[j, 1], gsize - 1 - selected_atoms[j, 0]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    ax = axs[2]
    ax.set_title('Array final sin defectos')
    ax.set_xlim(-1, gsize)
    ax.set_ylim(-1, gsize)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.scatter(final_targets[:, 1], gsize - 1 - final_targets[:, 0], c='red', s=20)
    
    plt.tight_layout()
    plt.show()

def _plot_slider(selected_atoms, final_targets, gsize, steps, discarded, t):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-1, gsize)
    ax.set_ylim(-1, gsize)
    ax.set_xticks(range(gsize))
    ax.set_yticks(range(gsize))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(f'Movimiento sin cruces\nTiempo: {t:.4f} s')
    scat = ax.scatter([], [], c='blue', s=20, zorder=3)
    ax.scatter(final_targets[:, 1], gsize - 1 - final_targets[:, 0], c='red', marker='x', s=30, zorder=1)
    disc_plot = ax.scatter([], [], c='gray', s=20, zorder=2)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Paso', 0, steps, valinit=0, valstep=1)
    
    def update(v):
        step = int(slider.val)
        tau = step / steps
        interp = (1 - tau) * selected_atoms + tau * final_targets
        scat.set_offsets(np.column_stack([interp[:, 1], gsize - 1 - interp[:, 0]]))
        if step == 0 and discarded.size:
            disc_plot.set_offsets(np.column_stack([discarded[:, 1], gsize - 1 - discarded[:, 0]]))
        else:
            disc_plot.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()

def main():
    print('Generando array inicial...')
    array = generate_random_array(GRID_SIZE, OCCUPANCY, SEED)
    atoms = get_atom_positions(array)
    targets = get_target_positions(GRID_SIZE, TARGET_SIZE)

    print(f'Átomos encontrados: {len(atoms)}')
    print(f'Posiciones objetivo: {len(targets)}')
    
    if len(atoms) < len(targets):
        print('No hay suficientes átomos para llenar el array sin defectos.')
        return

    print("Ejecutando asignación global (sin cruces)...")
    assigned_atoms, ordered_targets, assignment, t_assign = simple_global_assignment(atoms, targets, ALPHA)
    
    print(f"Tiempo total: {t_assign:.4f} s")
    
    discarded = find_discarded_atoms(atoms, assigned_atoms)
    _plot_rearrangement(atoms, assigned_atoms, ordered_targets, assignment, GRID_SIZE, t_assign)
    _plot_slider(assigned_atoms, ordered_targets, GRID_SIZE, STEPS, discarded, t_assign)

if __name__ == '__main__':
    main()