# Mandelbrot set in Python
---
### Goal
fast and easy to use mandelbrot renderings and animations in python

### Usage
most functions have their own `test*.py` file

testAll.py executes all test files.  
For the first test, many windows pop up and matplotlib is a bit slow, so be patient. The tests only continue once all windows have been closed.


### Backends
1. own complex numbers
2. built in complex numbers
3. numpy arrays
4. numba jit
5. tensorflow
6. CUDA

### Requirements
the backend falls back to the next best, but animations using anything else then CUDA or maybe numba are really slow.

always:
- matplotlib
- itertools
- numpy

optional:
- tensorflow
- numba
- CUDA from their website, CUDA compatible GPU
---
### State
Nothing is documented, maybe I'll do this one day 


Might work in jupyter/ colab  
Mandelbrot.ipynb works in colab, but is an old version.  
Enable GPU in google colab by changing runtime type to GPU