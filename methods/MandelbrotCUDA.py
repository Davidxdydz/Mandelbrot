from methods.Mandelbrot import Mandelbrot
from numba import vectorize
import numpy as np

# TODO expose float32 vs float64

@vectorize(['int32(float64,float64,int32)'], target='cuda')
def divergence(x, y, iterations):
    zr = 0
    zi = 0
    for i in range(iterations):
        zr, zi = zr*zr-zi*zi + x, 2 * zi*zr + y
        if (zr * zr + zi * zi) >= 4:
            return i
    return 0


class MandelbrotCUDA(Mandelbrot):
    def iterations_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        aspect_ratio = (x_max-x_min)/(y_max - y_min)
        width, height = self.get_pixel_size(aspect_ratio)
        xs = np.linspace(x_min, x_max, width, dtype=np.float64)
        ys = np.linspace(y_min, y_max,
                         height, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        return divergence(X, Y, self.max_iterations)
