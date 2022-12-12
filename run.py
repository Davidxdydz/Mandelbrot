from methods.Mandelbrot import Mandelbrot
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="")
parser.add_argument("--height", default=500, type=int)
args = parser.parse_args()

mandelbrot = Mandelbrot(max_iterations=1000,
                        backend=args.backend, height=args.height)
mandelbrot.interactive()
