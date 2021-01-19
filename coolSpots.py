from Mandelbrot import plotMandelbrot
from util.constants import params

for param in params[:-1]:
    plotMandelbrot(*param,1000,block = False)
plotMandelbrot(*params[-1],1000)