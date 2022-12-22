import numpy as np
from collections import OrderedDict
import logging
import matplotlib.pyplot as plt
from typing import Tuple
import time
import datetime


class Mandelbrot:

    available_backends = None
    # TODO move backends to manager class

    def __init__(self, max_iterations=100, height=500, backend=""):
        self.max_iterations = max_iterations
        self.height = height

    def __new__(cls, max_iterations=100, height=500, backend=""):
        backends = Mandelbrot.get_available_backends()
        if backend in backends:
            return super().__new__(backends[backend])
        elif backend == "":
            return super().__new__(next(iter(backends.values())))
        else:
            raise ValueError(f"Backend {backend} not found")

    @staticmethod
    def load_available_backends() -> OrderedDict:
        # TODO make this lazy
        available_backends = OrderedDict()
        # TODO order from fastest to slowest

        try:
            from methods.MandelbrotCUDA import MandelbrotCUDA
            available_backends["cuda"] = MandelbrotCUDA
        except ImportError:
            logging.warning(
                "Could not load MandelbrotCUDA. Try installing CUDA")

        try:
            from methods.MandelbrotNumba import MandelbrotNumba
            available_backends["numba"] = MandelbrotNumba
        except ImportError:
            logging.warning(
                "Could not load MandelbrotNumba. Try installing numba.")

        try:
            from methods.MandelbrotNumbaVectorized import MandelbrotNumbaVectorized
            available_backends["numbaVectorized"] = MandelbrotNumbaVectorized
        except ImportError:
            logging.warning(
                "Could not load MandelbrotNumbaVectorized. Try installing numba.")

        try:
            from methods.MandelbrotNumpy import MandelbrotNumpy
            available_backends["numpy"] = MandelbrotNumpy
        except ImportError:
            logging.warning(
                "Could not load MandelbrotNumpy. Try installing numpy.")

        try:
            from methods.MandelbrotPlain import MandelbrotPlain
            available_backends["plain"] = MandelbrotPlain
        except ImportError:
            logging.warning("Could not load MandelbrotPlain")

        try:
            from methods.MandelbrotPlainKomplex import MandelbrotPlainKomplex
            available_backends["plainKomplex"] = MandelbrotPlainKomplex
        except ImportError:
            logging.warning("Could not load MandelbrotPlainKomplex")
        return available_backends

    @staticmethod
    def get_available_backends() -> OrderedDict:
        if Mandelbrot.available_backends is None:
            Mandelbrot.available_backends = Mandelbrot.load_available_backends()
        return Mandelbrot.available_backends

    def iterations(self, x: float, y: float, zoom: float, aspect_ratio: float = 1) -> np.ndarray:
        """
        Calculate number of iterations needed for points around (`x`,`y`) to diverge. 

        Parameters
        ----------
            x : float
                center of returned image on the real axis
            y : float
                center of returned image on the imaginary axis
            zoom : float
                TODO
        """
        factor = 1/zoom
        return self.iterations_by_bounds(
            x-factor*aspect_ratio,
            x + factor * aspect_ratio,
            y-factor,
            y+factor,
            *self.get_pixel_size(aspect_ratio)
        )

    def get_pixel_size(self, aspect_ratio) -> Tuple[int, int]:
        width = int(self.height * aspect_ratio)
        return width, int(self.height)

    def timed_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[np.ndarray, datetime.timedelta]:
        start = time.perf_counter()
        result = self.iterations_by_bounds(x_min, x_max, y_min, y_max)
        end = time.perf_counter()
        return result, datetime.timedelta(seconds=end-start)

    def iterations_by_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        """
        Calculate number of iterations needed for points around (`x`,`y`) to diverge. 

        Parameters
        ----------
            x : float
                center of returned image on the real axis
            y : float
                center of returned image on the imaginary axis
            zoom : float
                TODO
        """
        raise NotImplementedError()

    def __call__(self, x: float = -0.5, y: float = 0, zoom: float = 0.9) -> np.ndarray:
        """
        Calculate number of iterations needed for points around (`x`,`y`) to diverge. 

        Parameters
        ----------
            x : float
                center of returned image on the real axis
            y : float
                center of returned image on the imaginary axis
            zoom : float
                TODO
        """
        return self.iterations(x, y, zoom)

    def interactive(self, cmap = "hot"):
        ax: plt.Axes = plt.gca()
        bounds = (-2.0, 1, -1.5, 1.5)
        im, t = self.timed_by_bounds(*bounds)
        image = ax.imshow(np.flip(im, 0),
                          extent=bounds, cmap=cmap, origin="upper")
        title = plt.title(
            f"{self.__class__.__name__}, {self.max_iterations} iterations in {t}")

        def on_ylims_change(axes: plt.Axes):
            nonlocal bounds
            x_min, x_max = axes.get_xlim()
            y_min, y_max = axes.get_ylim()
            new_bounds = (x_min, x_max, y_min, y_max)
            if new_bounds == bounds:
                return
            bounds = new_bounds
            im, t = self.timed_by_bounds(*bounds)
            image.set_array(np.flip(im, 0))
            image.set_extent(bounds)
            title.set_text(
                f"{self.__class__.__name__}, {self.max_iterations} iterations in {t}")
        ax.callbacks.connect('ylim_changed', on_ylims_change)
        plt.show()
