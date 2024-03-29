from Mandelbrot import plotMandelbrot
from util.constants import params
import numpy as np

filters = (
    None,
    lambda x: np.log(np.e + x) - 1,
    lambda x: np.sqrt(x),
    lambda x: x**2
)
cmaps = ['afmhot', 'jet', 'hot', 'cubehelix']

plotMandelbrot(params[1], 500, cmaps=cmaps,
               filters=filters, windowTitle="testFilters",block = False)
plotMandelbrot(params[0], 500, cmaps=cmaps,
              filters=filters, windowTitle="testFilters")
