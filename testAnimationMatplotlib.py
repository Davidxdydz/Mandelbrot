from Mandelbrot import animateZoom
from util.constants import params
import matplotlib.pyplot as plt

x, y, zoomMax, iterations = params[1]
anim = animateZoom(x, y, 1, zoomMax, iterations, 1000, fps=24, duration=5)
plt.show()
