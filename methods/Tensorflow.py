import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

def mandelbrot(x, y, zoom, iterations,yRes, aspectRatio=1):
    factor = 1/zoom
    width = int(yRes*aspectRatio)
    height = int(yRes)
    x = np.linspace(x-factor*aspectRatio, x+factor*aspectRatio, width)
    y = np.linspace(y-factor, y+factor, height)

    X, Y = np.meshgrid(x, y)
    coords = 1j*Y+X

    xs = tf.constant(coords.astype(np.complex64))
    zs = tf.Variable(tf.zeros_like(xs,np.complex64))
    not_diverged =tf.Variable(tf.zeros_like(xs, tf.bool))
    ns = tf.Variable(tf.zeros_like(xs, tf.int32))

    for i in range(iterations): 
      zs = zs*zs+xs
      not_diverged = tf.abs(zs) < 2
      ns = ns + tf.cast( not_diverged, tf.int32)
    n = ns.numpy()
    n[n == iterations] = 0
    return n