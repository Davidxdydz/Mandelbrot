from Mandelbrot import testBackends, forceCompile
from util.constants import params

forceCompile()
testBackends(-0.5, 0, 1, 10, 10, windowTitle="testBackends", block=False)
testBackends(*params[0], 100, 4/3, windowTitle="testBackends", block=False)
testBackends(*params[1], 50, windowTitle="testBackends", block=False)
testBackends(*params[1], 1000, functions=['cuda',
                                          'tf', 'numba'], windowTitle="testBackends")
