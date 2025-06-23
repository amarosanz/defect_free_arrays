import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Rasterizer:
    def __init__(self, 
                 resolution : tuple = (20, 20), # Resolution of the rasterized image
                 threshold: int = 128, # Threshold for binarization (0-255)
                 show: bool = True, # Whether to display the rasterized image
                 separation: int = 0): # Separation between pixels in the rasterized image
        
        self.resolution = resolution 
        self.threshold = threshold
        self.show = show
        self.separation = separation

    def rasterize_image(self, image_path):
        # Open the image and convert it to grayscale
        img = Image.open(image_path).convert('L')
        # Resize the image to the specified resolution
        img_reduced = img.resize(self.resolution, Image.NEAREST)
        # Convert the image to a binary matrix based on the threshold
        img_array = np.array(img_reduced)
        binary_matrix = (img_array > self.threshold).astype(int)

        # If separation is specified, create a new matrix with the specified separation
        # between pixels in the rasterized image
        if self.separation > 0:
            h, w = binary_matrix.shape
            new_matrix = np.ones((h + (h - 1) * self.separation, w + (w - 1) * self.separation), dtype=int)

            for i in range(h):
                for j in range(w):
                    new_matrix[i * (1 + self.separation), j * (1 + self.separation)] = binary_matrix[i, j]

            binary_matrix = new_matrix

        # Show the rasterized image if specified
        if self.show:
            plt.figure(figsize=(8, 8))
            plt.imshow(binary_matrix, cmap='gray', interpolation='none')
            plt.axis('off')
            plt.title(f'Rasterized Image ({self.resolution}, sep={self.separation})')
            plt.show()

        return binary_matrix




# Ejemplo de uso
if __name__ == "__main__":
    rasterizer = Rasterizer(resolution=(200, 200), threshold=128, show=True, separation=1)
    binary_matrix = rasterizer.rasterize_image('/home/amaro/Downloads/buho.jpg')
    print("Binary Matrix:")
    print(binary_matrix)

