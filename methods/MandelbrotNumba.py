from methods.Mandelbrot import Mandelbrot
from numba import jit
import numpy as np

# TODO expose float32 vs float64


@jit(nopython=True)
def divergence(x, y, iterations):
    zr = 0
    zi = 0
    for i in range(iterations):
        zr, zi = zr*zr-zi*zi + x, 2 * zi*zr + y
        if (zr * zr + zi * zi) >= 4:
            return i
    return 0


@jit(nopython=True)
def iterations_by_bounds(x_min, x_max, y_min, y_max, width, height, max_iterations):

    n = np.zeros((height, width), dtype=np.uint32)

    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_min, y_max, height)

    for xi, real in zip(range(width), xs):
        for yi, imag in zip(range(height), ys):
            n[yi, xi] = divergence(real, imag, max_iterations)

    return n


class MandelbrotNumba(Mandelbrot):
    def iterations_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        aspect_ratio = (x_max-x_min)/(y_max-y_min)
        width, height = self.get_pixel_size(aspect_ratio)
        return iterations_by_bounds(x_min, x_max, y_min, y_max, width, height, self.max_iterations)
