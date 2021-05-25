from typing import Dict, Tuple, Callable, List
from itertools import product

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import jax.numpy as jnp


def plot_convergence(
        data: Dict,
        loss_fn: Callable,
        eq_constr: List[Callable],
):
    # plot contours
    Nx, Ny = 40, 40
    x = np.linspace(-2, 2, Nx)
    y = np.linspace(-2, 2, Ny)
    xx, yy = np.meshgrid(x, y)

    zz = loss_fn(xx, yy)

    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    plt.xlabel(r'x')
    plt.ylabel(r'y')

    plt.contour(xx, yy, zz, 30, cmap='jet')

    # plot constraints
    for c_fn in eq_constr:
        c_vals_x = c_fn(xx, yy)
        plt.contour(xx, yy, c_vals_x, [0], colors='k')

    # path
    path = np.array([np.array(item.x) for item in data.values()])
    plt.plot(path[:, 0], path[:, 1], marker='o', color='red')

    return ax


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
