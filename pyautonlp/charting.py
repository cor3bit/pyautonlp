from typing import Dict, Tuple, Callable
from itertools import product

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import jax.numpy as jnp


def plot_convergence(data: Dict, loss_fn: Callable):
    # xmin, xmax, xstep = -4.5, 4.5, .2
    # ymin, ymax, ystep = -4.5, 4.5, .2
    #
    # x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))

    # x, y
    Nx, Ny = 10, 10
    x = np.linspace(-2, 2, Nx)
    y = np.linspace(-2, 2, Ny)
    xx, yy = np.meshgrid(x, y)

    zz = loss_fn(xx, yy)

    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    plt.xlabel(r'x')
    plt.ylabel(r'y')

    a = 1
    # z = loss_fn(ndata)

    plt.contour(xx, yy, zz, 25, cmap='jet')

    plt.show()


def plot_training_loss(data: Dict):
    # (curr_x, curr_m, loss, step_size, conv_penalty)
    times = np.fromiter(data.keys(), dtype=float)
    losses = np.array([item.loss for item in data.values()])
    return sns.lineplot(x=times, y=losses)


def plot_penalty(data: Dict):
    # (curr_x, curr_m, loss, step_size, conv_penalty)
    times = np.fromiter(data.keys(), dtype=float)
    penalties = np.array([item.penalty for item in data.values()])
    return sns.lineplot(x=times, y=np.log(penalties))


def plot_alpha(data: Dict):
    # (curr_x, curr_m, loss, step_size, conv_penalty)
    times = np.fromiter(data.keys(), dtype=float)
    alphas = np.array([item.alpha for item in data.values()])
    return sns.lineplot(x=times[:-1], y=alphas[:-1])
