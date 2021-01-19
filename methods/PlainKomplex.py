from .Komplex import Komplex

def mandelbrot(x, y, zoom, iterations,yRes, aspectRatio=1):

    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)

    n = [[0 for x in range(width)] for y in range(height)]
    left = x-factor*aspectRatio
    bottom = y-factor

    for y in range(height):
        for x in range(width):
            real = left + x/(width-1)*factor*2*aspectRatio
            imag = bottom + y/(height-1)*factor*2
            z = Komplex(0, 0)
            c = Komplex(real, imag)
            for count in range(iterations):
                z = z * z + c
                if abs(z) >= 2.0:
                    n[y][x] = count
                    break
    return n
