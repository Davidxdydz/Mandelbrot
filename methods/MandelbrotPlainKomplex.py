from methods.Mandelbrot import Mandelbrot
from methods.Komplex import Komplex


class MandelbrotPlainKomplex(Mandelbrot):
    def iterations_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        aspect_ratio = (x_max-x_min)/(y_max-y_min)
        width, height = self.get_pixel_size(aspect_ratio)
        n = [[0 for x in range(width)]
             for y in range(height)]
        left = x_min
        bottom = y_min
        x_step = (x_max-x_min)/(width-1)
        y_step = (y_max-y_min)/(height-1)

        for y in range(height):
            for x in range(width):
                real = left + x*x_step
                imag = bottom + y*y_step
                z = Komplex(0, 0)
                c = Komplex(real, imag)
                for count in range(self.max_iterations):
                    z = z * z + c
                    if abs(z) >= 2.0:
                        n[y][x] = count
                        break
        return n
