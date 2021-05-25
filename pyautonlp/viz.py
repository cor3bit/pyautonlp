from typing import Dict, Tuple, Callable, List
from itertools import product

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import jax.numpy as jnp


class Visualizer:
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            solver_caches: List[Dict] = None,
            cache_names: List[str] = None,
            x1_bounds: Tuple[float, float] = (-1, 1),
            x2_bounds: Tuple[float, float] = (-1, 1),
            separate: bool = False,
    ):
        self._loss_fn = loss_fn
        self._eq_constr = eq_constr
        self._ineq_constr = ineq_constr
        self._solver_caches = solver_caches
        self._cache_names = cache_names
        self._x1_bounds = x1_bounds
        self._x2_bounds = x2_bounds
        self._separate = separate

    def plot_convergence(self):
        # create numpy meshgrid
        Nx, Ny = 40, 40
        x1_xaxis, x1_yaxis = self._x1_bounds
        x2_xaxis, x2_yaxis = self._x2_bounds
        x = np.linspace(x1_xaxis, x1_yaxis, Nx)
        y = np.linspace(x2_xaxis, x2_yaxis, Ny)
        xx, yy = np.meshgrid(x, y)

        # f(x) values
        zz = self._loss_fn(xx, yy)

        # plot contours of f(x)
        # fig = plt.figure()
        ax = plt.axes(xlim=(x1_xaxis, x1_yaxis), ylim=(x2_xaxis, x2_yaxis))
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        plt.contour(xx, yy, zz, 30, cmap='jet')

        # plot constraints
        for c_fn in self._eq_constr:
            cvals = c_fn(xx, yy)
            plt.contour(xx, yy, cvals, [0], colors='k')

        # plot optimizer path
        if self._solver_caches:
            if not self._separate:
                for cache, name in zip(self._solver_caches, self._cache_names):
                    path = np.array([np.array(item.x) for item in cache.values()])
                    plt.plot(path[:, 0], path[:, 1], marker='o', label=name, color='red')
            else:
                raise NotImplementedError

        # ax.set_title('Convergence')
        plt.legend()
        plt.show()

    # def plot_training_loss(self):
    #     # (curr_x, curr_m, loss, step_size, conv_penalty)
    #     times = np.fromiter(data.keys(), dtype=float)
    #     losses = np.array([item.loss for item in data.values()])
    #     return sns.lineplot(x=times, y=losses)
    #
    # def plot_penalty(self):
    #     # (curr_x, curr_m, loss, step_size, conv_penalty)
    #     times = np.fromiter(data.keys(), dtype=float)
    #     penalties = np.array([item.penalty for item in data.values()])
    #     return sns.lineplot(x=times, y=np.log(penalties))
    #
    # def plot_alpha(self):
    #     # (curr_x, curr_m, loss, step_size, conv_penalty)
    #     times = np.fromiter(data.keys(), dtype=float)
    #     alphas = np.array([item.alpha for item in data.values()])
    #     return sns.lineplot(x=times[:-1], y=alphas[:-1])


# --------------- runner ---------------

if __name__ == '__main__':
    from pyautonlp.constr.constr_newton import CacheItem


    def loss(x, y):
        # min z = 0.5 * x_T * x + 1_T * x
        return 0.5 * x * x + 0.5 * y * y + x + y


    def constr(x, y):
        return x * x + y * y - 1


    fake_caches = [
        {
            0: CacheItem(x=[-1, -1], m=[2.], loss=1., alpha=0.1, penalty=3., sigma=1.),
            1: CacheItem(x=[-0.5, -0.5], m=[2.], loss=0.5, alpha=0.1, penalty=3., sigma=1.),
            2: CacheItem(x=[-0.3, -0.3], m=[2.], loss=0.4, alpha=0.05, penalty=2., sigma=1.),
            3: CacheItem(x=[1, -0.3], m=[2.], loss=0., alpha=0.03, penalty=2., sigma=1.),
            4: CacheItem(x=[0.7, 0.7], m=[2.], loss=0., alpha=0.01, penalty=1., sigma=1.),
        },
        {
            0: CacheItem(x=[1, 1], m=[2.], loss=1., alpha=0.1, penalty=3., sigma=1.),
            1: CacheItem(x=[0.5, 0.5], m=[2.], loss=0.5, alpha=0.1, penalty=3., sigma=1.),
            2: CacheItem(x=[0.3, 0.3], m=[2.], loss=0.4, alpha=0.05, penalty=2., sigma=1.),
            3: CacheItem(x=[0.2, 0.2], m=[2.], loss=0., alpha=0.03, penalty=2., sigma=1.),
            4: CacheItem(x=[0.1, 0.1], m=[2.], loss=0., alpha=0.01, penalty=1., sigma=1.),
        },
    ]

    visualizer = Visualizer(
        loss_fn=loss,
        eq_constr=[constr],
        solver_caches=fake_caches,
        cache_names=['temp1', 'temp2'],
        x1_bounds=(-2, 2),
        x2_bounds=(-2, 2),
    )

    visualizer.plot_convergence()
