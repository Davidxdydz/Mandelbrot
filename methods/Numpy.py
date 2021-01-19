import numpy as np


def mandelbrot(x, y, zoom, iterations, yRes, aspectRatio=1):

    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)
    x = np.linspace(x-factor*aspectRatio, x+factor*aspectRatio, width)
    y = np.linspace(y-factor, y+factor, height)

    X, Y = np.meshgrid(x, y)
    coords = 1j*Y+X

    z = np.zeros(coords.shape, dtype=np.complex64)
    n = np.zeros((height, width), dtype=np.int32)

    for i in range(iterations+1):
        mask = np.less(np.abs(z), 2)
        z[mask] = z[mask]**2+coords[mask]
        n[mask] = i
    n[n == iterations] = 0
    return n
