# Mandelbrot set in Python
---
### Goal
fast and easy to use mandelbrot renderings and animations in python

### Usage
Just run `run.py`.


### Backends
1. own complex numbers
2. built in complex numbers
3. numpy arrays
4. numba jit
5. CUDA

### Requirements
the backend falls back to the next best, but animations using anything else then CUDA or maybe numba are really slow.

always:
- matplotlib
- numpy

optional:
- numba
- CUDA from their website, CUDA compatible GPU
---
### Current State
works, but rewriting everything because some of the current code is ugly.