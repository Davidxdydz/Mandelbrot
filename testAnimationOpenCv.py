from Mandelbrot import animateZoomCv2
from util.constants import params
import matplotlib.pyplot as plt

x,y,zoomMax,iterations = params[1]
animateZoomCv2(x,y,1,zoomMax,iterations,1000,fps = 24,duration = 5)