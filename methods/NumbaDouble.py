from numba import vectorize
import numpy as np

@vectorize(['int32(float64,float64,int32)'])
def divergence(x, y, iterations):
    zr = 0
    zi = 0
    for i in range(iterations):
        zr, zi = zr*zr-zi*zi + x, 2 * zi*zr + y
        if (zr * zr + zi * zi) >= 4:
            return i
    return 0

def mandelbrot(x, y, zoom, iterations,yRes, aspectRatio=1):
    # this is actually way faster than numpy

    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)


    xs = np.linspace(x-factor*aspectRatio, x+factor*aspectRatio, width,dtype = np.float64)
    ys = np.linspace(y-factor, y+factor, height,dtype = np.float64)
    X, Y = np.meshgrid(xs, ys)

    return divergence(X,Y,iterations)
