from numba import jit
import numpy as np

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
def mandelbrot(x, y, zoom, iterations,yRes, aspectRatio=1):
    # this is actually way faster than numpy

    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)

    n = np.zeros((height, width), dtype=np.uint16)

    xs = np.linspace(x-factor*aspectRatio, x+factor*aspectRatio, width)
    ys = np.linspace(y-factor, y+factor, height)

    for xi, real in zip(range(width), xs):
        for yi, imag in zip(range(height), ys):
            n[yi, xi] = divergence(real, imag, iterations)

    return n

@jit(nopython=True)
def divergesJulia(x, y,cr,ci, iterations):
    zr = x
    zi = y
    for i in range(iterations):
        zr, zi = zr*zr-zi*zi + cr, 2 * zi*zr + ci
        if (zr * zr + zi * zi) >= 4:
            return i
    return 0


@jit(nopython=True)
def julia(x, y,cr,ci, zoom, yRes, iterations, aspectRatio=1):
    # this is actually way faster than numpy

    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)

    n = np.zeros((height, width), dtype=np.uint16)

    xs = np.linspace(x-factor*aspectRatio, x+factor*aspectRatio, width)
    ys = np.linspace(y-factor, y+factor, height)

    for xi, real in zip(range(width), xs):
        for yi, imag in zip(range(height), ys):
            n[yi, xi] = divergesJulia(real, imag,cr,ci, iterations)

    return n
