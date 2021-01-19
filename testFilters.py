from Mandelbrot import plotMandelbrot
from util.constants import params
import numpy as np

filters = (
    None,
    lambda x: np.log(x),
    lambda x: np.sqrt(x),
    lambda x: x**2
)
cmaps = ['afmhot', 'jet', 'hot', 'cubehelix']

plotMandelbrot(*params[1], 500, cmaps=cmaps,
               filters=filters, windowTitle="testFilters")
