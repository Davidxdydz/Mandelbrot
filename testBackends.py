from Mandelbrot import testBackends, forceCompile
from util.constants import params

forceCompile()
testBackends(-0.5, 0, 1, 10, 10)
testBackends(*params[0], 300, 4/3)
testBackends(*params[1], 50)
testBackends(*params[1], 1000, functions=['cuda', 'numpy', 'tf', 'numba'])
