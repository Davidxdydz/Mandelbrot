from methods.Mandelbrot import Mandelbrot
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="")
parser.add_argument("--height", default=1000, type=int)
parser.add_argument("--iterations", default=3000, type=int)
parser.add_argument("--cmap", default="hot", type=str)
args = parser.parse_args()

mandelbrot = Mandelbrot(max_iterations=args.iterations,
                        backend=args.backend, height=args.height)
mandelbrot.interactive(cmap=args.cmap)
