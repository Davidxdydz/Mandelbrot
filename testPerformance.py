import numpy as np
from Mandelbrot import backends, forceCompile
from tqdm.auto import tqdm
from util.timer import timer
import matplotlib.pyplot as plt
from util.constants import params

ress = np.logspace(1, 3.5, base=10, num=40)
times = {key: [] for key in backends.keys()}

forceCompile()

x = -0.745428
y = 0.113009
zoom = 33333
iterations = 500

for res in tqdm(ress):
    for key, f in backends.items():
        if times[key] and times[key][-1] > 1:
            continue
        _, t = timer(f, *params[1], res)
        times[key].append(t)

for key, vals in times.items():
    plt.plot(ress[:len(vals)], vals, label=key)

plt.legend()
plt.ylabel("log(t) [s]")
plt.xlabel("log(yRes)")
plt.yscale("log")
plt.xscale("log")
plt.gcf().canvas.set_window_title('testPerformance')
plt.title("time per resolution (double logarithmic)")
plt.show(block=False)
plt.figure()

for key, vals in times.items():
    plt.plot(ress[:len(vals)], vals, label=key)
plt.legend()
plt.ylabel("t [s]")
plt.xlabel("yRes")
plt.gcf().canvas.set_window_title('testPerformance')
plt.title("time per resolution (linear)")
plt.show()
