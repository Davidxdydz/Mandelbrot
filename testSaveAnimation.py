from Mandelbrot import animateZoom
from util.constants import params
import matplotlib.animation as animation
import os

filename = "animation.mp4"

x, y, zoomMax, iterations = params[1]
anim = animateZoom(x, y, 1, zoomMax, iterations, 1000, fps=30, duration=10)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, bitrate=8000)

anim.save(filename,writer=writer)

print("Saved to",os.path.abspath(filename))