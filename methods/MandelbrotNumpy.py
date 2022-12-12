from methods.Mandelbrot import Mandelbrot
import numpy as np


class MandelbrotNumpy(Mandelbrot):
    def iterations_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        aspect_ratio = (x_max-x_min)/(y_max-y_min)
        width, height = self.get_pixel_size(aspect_ratio)
        x = np.linspace(x_min, x_max, width)
        y = np.linspace(y_min, y_max, height)

        X, Y = np.meshgrid(x, y)
        coords: np.ndarray = 1j*Y+X

        z = np.zeros_like(coords, dtype=np.complex64)
        n = np.zeros_like(coords, dtype=np.int32)

        for i in range(self.max_iterations):
            mask = np.less(np.abs(z), 2)
            z[mask] = z[mask]**2+coords[mask]
            n[mask] = i
        n[n == self.max_iterations-1] = 0
        return n
