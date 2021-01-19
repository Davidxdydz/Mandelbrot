from itertools import product
from google.protobuf import message
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import matplotlib.cm as cm
from numpy.lib.shape_base import take_along_axis
from util.timer import timer
from collections import OrderedDict

backends = OrderedDict()

try:
    import methods.CUDA
    backends['cuda'] = methods.CUDA.mandelbrot
except Exception as e:
    print(e)

try:
    import methods.Numba
    backends['numba'] = methods.Numba.mandelbrot
except Exception as e:
    print(e)

try:
    import methods.Tensorflow
    backends['tf'] = methods.Tensorflow.mandelbrot
except Exception as e:
    print(e)

try:
    import methods.Numpy
    backends['numpy'] = methods.Numpy.mandelbrot
except Exception as e:
    print(e)

try:
    import methods.Plain
    backends['complex'] = methods.Plain.mandelbrot
except Exception as e:
    print(e)

try:
    import methods.PlainKomplex
    backends['komplex'] = methods.PlainKomplex.mandelbrot
except Exception as e:
    print(e)


def remapCmap(image, filter=None):
    if filter:
        return filter(image)

    return image


def plotImage(images, cmaps=("hot",), filters=(None,), titles=(None,), windowTitle=None, **kwargs):
    images = np.array(images)
    titles = np.atleast_1d(titles)
    if images.ndim == 2:
        images = np.array([images])
    toPlot = list(product(cmaps, enumerate(filters),images))
    sideLength = int(np.ceil(np.sqrt(len(toPlot))))
    (height, width) = images[0].shape
    aspectRatio = width/height
    fig, axes = plt.subplots(sideLength, sideLength, figsize=(
        aspectRatio*10, 10), squeeze=False, sharey=True, sharex=True)
    if titles:
        fig.suptitle(titles[0])
    for xi in range(sideLength):
        for yi in range(sideLength):
            index = yi*sideLength + xi
            if index >= len(toPlot):
                break
            cmap, (n, filter),image = toPlot[index]
            axes[yi, xi].imshow(remapCmap(image, filter), cmap=cmap)
            axes[yi, xi].set_title(f'"{cmap}", filter {n}')
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    plt.show(**kwargs)


def testBackends(x, y, zoom, iterations, yRes, aspectRatio=1, cmap='hot', title=None, functions=None, filter=None, windowTitle=None, **kwargs):
    ml = list(backends.items())
    if functions:
        ml = [(key, backends[key]) for key in functions if key in backends]
    prevImage = None
    prevKey = ""
    sideLength = int(np.ceil(np.sqrt(len(ml))))
    fig, axes = plt.subplots(sideLength, sideLength, figsize=(
        aspectRatio*10, 10), squeeze=False, sharey=True, sharex=True)
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(
            f"x:{x}, y:{y}, zoom:{zoom:.2f}, {iterations} iterations, cmap={cmap}")
    for xi in range(sideLength):
        for yi in range(sideLength):
            index = yi*sideLength + xi
            if index >= len(ml):
                break
            key, f = ml[index]
            image, t = timer(f, x, y, zoom, iterations, yRes, aspectRatio)
            image = remapCmap(image, filter)
            if index == 0:
                prevImage = image
                prevKey = key
            msg = ""
            if not np.array_equal(prevImage, image):
                msg = f"\ndiffers from {prevKey}"
            prevImage = image
            prevKey = key
            axes[yi, xi].imshow(image, cmap=cmap)
            axes[yi, xi].set_title(f'{key}: {t:.3f}s{msg}')
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    plt.show(**kwargs)


def plotMandelbrot(coords, yRes, aspectRatio=1, cmaps=("hot",), title=None, subtitle=None, backend=None, filters=(None,), windowTitle=None, **kwargs):
    # kwargs for plt.show(**kwargs)
    coords = np.atleast_2d(coords)
    x, y, zoom, iterations = coords[0]
    if backend is None:
        backend = list(backends.keys())[0]
    if not backend in backends:
        print(
            f'"{backend}" not found, available backends:{",".join(backends.keys())}')
        return
    count, t = timer(backends[backend], x, y, zoom,
                     iterations, yRes, aspectRatio)
    if not title:
        title = f"x:{x}, y:{y}, zoom:{zoom:.2f}, {iterations} iterations ({t:.3f}s)"
    if subtitle:
        title = title + "\n" + subtitle
    plotImage(count, cmaps, titles=title, filters=filters,
              windowTitle=windowTitle, **kwargs)


def animateZoom(x, y, zoomMin, zoomMax, iterations, yRes, aspectRatio=1, cmap="afmhot", speed=0.1, fps=2, duration=5, backend=None, filter=None):

    if backend is None:
        backend = list(backends.keys())[0]

    if not backend in backends:
        print(
            f'"{backend}" not found, available backends:{",".join(backends.keys())}')
        return
    fig = plt.figure(figsize=(int(aspectRatio*10), 10))
    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dt = 1/fps * 1000  # in ms

    count = backends[backend](x, y, zoomMin, iterations, yRes, aspectRatio)
    count = remapCmap(count, filter)
    image = plt.imshow(count, cmap=cmap, animated=True)

    zooms = list(np.logspace(np.log(zoomMin)/np.log(1+speed), np.log(zoomMax) /
                             np.log(1+speed), base=1+speed, num=int(duration*fps)))

    def update(zoom):
        print(f" {zooms.index(zoom)+1}/{len(zooms)}", end="\r")
        count = backends[backend](x, y, zoom, iterations, yRes, aspectRatio)
        count = remapCmap(count, filter)
        image.set_array(count)
        return image,

    # TODO: tqdm on zooms is broken?? remove print from update
    return FuncAnimation(fig, update, zooms, interval=dt, blit=True)


cv2Available = False
try:
    import cv2
    cv2Available = True
except Exception as e:
    print(e)


import copy
def animateZoomCv2(x, y, zoomMin, zoomMax, iterations, yRes, aspectRatio=1, cmap="afmhot", speed=0.1, fps=30, duration=10, backend=None):
    if not cv2Available:
        print("cv2 not imported")
        return
    if backend is None:
        backend = list(backends.keys())[0]
    if not backend in backends:
        print(
            f'"{backend}" not found, available backends:{",".join(backends.keys())}')
        return
    dt = 1/fps * 1000
    cmap = copy.copy(cm.get_cmap(cmap))
    # force to return bgr instead of rgba
    cmap._init()
    cmap._lut = cmap._lut[..., [2, 1, 0]]
    frames = int(duration*fps)
    zooms = np.logspace(np.log(zoomMin)/np.log(1+speed),
                        np.log(zoomMax)/np.log(1+speed), base=1+speed, num=frames)
    bufferedFrames = [None]*frames

    def getFrame(i):
        if bufferedFrames[i] is None:
            zoom = zooms[i]
            image = backends[backend](
                x, y, zoom, iterations, yRes, aspectRatio)
            bufferedFrames[i] = cmap(image/image.max(), bytes=True)
        return bufferedFrames[i]

    def update(i):
        start = time.time()
        image = getFrame(i)
        cv2.imshow('Quit with Q', image)
        end = time.time()
        return max(int(dt-(end-start)*1000), 1)

    i = 0
    while(True):
        wait = update(i)
        if cv2.waitKey(wait) & 0xff == ord('q'):
            break
        i = (i+1) % frames
    cv2.destroyAllWindows()


def forceCompile(output=False):
    if output:
        print("forcing the jit compiler...")
    for key, f in backends.items():
        if output:
            print(f"{key}...")
        f(0, 0, 1, 1, 3, 1)
    if output:
        print("done")
