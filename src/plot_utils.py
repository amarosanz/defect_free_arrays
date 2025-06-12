import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, data=None):
        self.data = data

    def plot_atom_positions(self, positions, title="Atom Positions"):
        """
        Dibuja un gráfico de dispersión de las posiciones de los átomos.
        
        Parameters:
            positions: Lista de posiciones de los átomos.
            title: Título del gráfico.
        """
        plt.scatter(positions[:, 0], positions[:, 1], c='blue')
        plt.title(title)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.show()

    def plot_interpolation_slider(self, values, slider_steps=10):
        """
        Muestra un gráfico con un slider para interpolación.
        
        Parameters:
            values: Lista de valores a interpolar.
            slider_steps: Número de pasos del slider (por defecto 10).
        """
        from matplotlib.widgets import Slider
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        
        slider_ax = fig.add_axes([0.25, 0.01, 0.65, 0.03])
        slider = Slider(slider_ax, 'Interpolation', 0, slider_steps, valinit=0)
        
        def update(val):
            idx = int(slider.val)
            ax.clear()
            ax.plot(values[idx])  # Aquí puedes usar el valor interpolado
            ax.set_title(f"Interpolation Step {idx}")
            plt.draw()

        slider.on_changed(update)
        plt.show()

