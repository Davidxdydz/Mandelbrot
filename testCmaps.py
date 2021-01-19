from Mandelbrot import plotMandelbrot
from util.constants import goodCmaps, allCmaps, params
import random

for i in range(len(goodCmaps)//4):
    plotMandelbrot(*params[1], 1000, cmaps=goodCmaps[i*4:i*4+4],
                   subtitle=f"{i*4}-{i*4+4} from goodCmaps", windowTitle="testCmaps", block=False)
    plotMandelbrot(*params[0], 1000, cmaps=goodCmaps[i*4:i*4+4],
                   subtitle=f"{i*4}-{i*4+4} from goodCmaps", windowTitle="testCmaps", block=False)


choices = random.choices(allCmaps, k=9)
plotMandelbrot(*params[1], 1000, cmaps=choices,
               subtitle="random from allCmaps", windowTitle="testCmaps", block=False)
plotMandelbrot(*params[0], 1000, cmaps=choices,
               subtitle="random from allCmaps", windowTitle="testCmaps")
