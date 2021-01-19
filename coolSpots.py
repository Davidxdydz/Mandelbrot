from Mandelbrot import plotMandelbrot
from util.constants import params
import numpy as np

for param in params[:-1]:
    plotMandelbrot(param, 1000, block=False, windowTitle="coolSpots")
plotMandelbrot(params[-1], 1000, windowTitle="coolSpots")
