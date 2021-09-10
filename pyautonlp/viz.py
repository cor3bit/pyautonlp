from typing import Dict, Tuple, Callable, List, Optional
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
            loss_fn: Callable = None,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            solver_type: Optional[str] = None,
            save_dir: Optional[str] = None,
            solver_caches: List[Dict] = None,
            cache_names: List[str] = None,
            x1_bounds: Tuple[float, float] = (-1, 1),
            x2_bounds: Tuple[float, float] = (-1, 1),
            n: int = 20,
            with_alpha_plot: bool = False,
            with_loss_plot: bool = False,
            with_penalty_plot: bool = False,
            separate: bool = False,
    ):
        self._loss_fn = loss_fn
        self._eq_constr = eq_constr
        self._ineq_constr = ineq_constr
        self._solver_caches = solver_caches
        self._cache_names = cache_names
        self._x1_bounds = x1_bounds
        self._x2_bounds = x2_bounds
        self._with_alpha_plot = with_alpha_plot
        self._with_loss_plot = with_loss_plot
        self._with_penalty_plot = with_penalty_plot
        self._separate = separate
        self._n = n
        self._solver_type = solver_type
        self._save_dir = save_dir

    def visualize(self):
        self.plot_convergence()

        if self._with_alpha_plot:
            self.plot_alpha()

        if self._with_loss_plot:
            self.plot_training_loss()

        if self._with_penalty_plot:
            self.plot_penalty()

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
        if self._eq_constr is not None:
            for c_fn in self._eq_constr:
                cvals = c_fn(xx, yy)
                plt.contour(xx, yy, cvals, [0], colors='k')

        if self._ineq_constr is not None:
            for c_fn in self._ineq_constr:
                cvals = c_fn(xx, yy)
                plt.contour(xx, yy, cvals, [0], colors='k')

        # plot optimizer path
        if self._solver_caches:
            if not self._separate:
                for cache, name in zip(self._solver_caches, self._cache_names):
                    path = np.array([np.array(item['x']) for item in cache.values()])
                    plt.plot(path[:, 0], path[:, 1], marker='o', label=name, color='red')
            else:
                raise NotImplementedError

        # ax.set_title('Convergence')
        plt.legend()
        plt.show()

    def plot_training_loss(self):
        if self._solver_caches:
            for cache, name in zip(self._solver_caches, self._cache_names):
                times = np.fromiter(cache.keys(), dtype=float)
                losses = np.array([item['loss'] for item in cache.values()])
                sns.lineplot(x=times, y=losses, label=name).set_title('Loss(t)')

        # ax.set_title('Convergence')
        plt.legend()
        plt.show()

    def plot_penalty(self):
        if self._solver_caches:
            for cache, name in zip(self._solver_caches, self._cache_names):
                times = np.fromiter(cache.keys(), dtype=float)
                penalties = np.array([item['penalty'] for item in cache.values()])
                sns.lineplot(x=times, y=penalties, label=name).set_title('KKT Penalty(t)')

        # ax.set_title('Convergence')
        plt.legend()
        plt.show()

    def plot_alpha(self):
        if self._solver_caches:
            for cache, name in zip(self._solver_caches, self._cache_names):
                times = np.fromiter(cache.keys(), dtype=float)
                alphas = np.array([item['alpha'] for item in cache.values()])
                sns.lineplot(x=times[:-1], y=alphas[:-1], label=name).set_title('Alpha(t)')

        plt.legend()
        plt.show()

    def plot_shooting(self):
        assert self._solver_caches is not None
        assert len(self._solver_caches) == 1
        cache = self._solver_caches[0]

        # extract meta data from cache
        xf = cache[0]['xf']
        if xf[0] == 2.:
            scenario = 'a'
        elif xf[0] == 1.:
            scenario = 'c'
        else:
            scenario = 'b'
        n_steps = cache[0]['n_steps']

        # 1) plot main time plot k vs. |g(x)|, |dLagrangian|, alpha
        fig, ax = plt.subplots()
        ax.set_yscale('log')

        times = np.fromiter(cache.keys(), dtype=float)
        vals = np.array([(
            step_cache['alpha'], step_cache['max_c_eq'],
            step_cache['max_grad_Lagrangian'], step_cache['loss'],
        ) for step_cache in cache.values()])

        ax.plot(times, vals[:, 1], '-', label=r'$\left\Vert g\right\Vert _{\infty}$')
        ax.plot(times, vals[:, 2], '-', label=r'$\left\Vert \nabla_{x}\mathcal{L}\right\Vert _{\infty}$')
        ax.plot(times, vals[:, 0], '-', label=r'$\alpha$')
        # ax.plot(times, vals[:, 3], '-', label=r'$Loss$')

        plt.legend()
        plt.xlabel('k')
        plt.show()

        if self._save_dir is not None:
            fig.savefig(self._save_dir + f'{self._solver_type}_{scenario}_n{n_steps}_main.png')

        # 2) plot w_k every N iterations
        t0, tf = 0., 2.
        u_min, u_max = -20., 20.
        N = self._n
        for k, step_cache in cache.items():
            if k % N == 0:
                fig, ax = plt.subplots()

                w_k = np.array(step_cache['u'])
                w_k_dims = w_k.shape[0]

                times = np.linspace(t0, tf, w_k_dims + 1)
                w_k_steps = np.concatenate((w_k, [w_k[-1]]))

                ax.plot(times, np.array([u_min for i in range(len(times))]), color='red', ls='--')
                ax.plot(times, np.array([u_max for i in range(len(times))]), color='red', ls='--')
                ax.step(times, w_k_steps, where='post')

                plt.title(f'SQP iteration: {k}')
                # plt.legend()
                plt.xlabel('$t$')
                plt.ylabel('$u$')
                plt.show()

                if self._save_dir is not None:
                    fig.savefig(self._save_dir + f'{self._solver_type}_{scenario}_n{n_steps}_i{k}.png')

        # 3) create gif of w_k evolution
        # TODO


# --------------- runner ---------------

if __name__ == '__main__':
    def loss(x, y):
        # min z = 0.5 * x_T * x + 1_T * x
        return 0.5 * x * x + 0.5 * y * y + x + y


    def constr(x, y):
        return x * x + y * y - 1


    fake_caches = [
        {
            0: dict(x=[-1, -1], m=[2.], loss=1., alpha=0.1, penalty=3., sigma=1.),
            1: dict(x=[-0.5, -0.5], m=[2.], loss=0.5, alpha=0.1, penalty=3., sigma=1.),
            2: dict(x=[-0.3, -0.3], m=[2.], loss=0.4, alpha=0.05, penalty=2., sigma=1.),
            3: dict(x=[1, -0.3], m=[2.], loss=0., alpha=0.03, penalty=2., sigma=1.),
            4: dict(x=[0.7, 0.7], m=[2.], loss=0., alpha=0.01, penalty=1., sigma=1.),
        },
        {
            0: dict(x=[1, 1], m=[2.], loss=1., alpha=0.1, penalty=3., sigma=1.),
            1: dict(x=[0.5, 0.5], m=[2.], loss=0.5, alpha=0.1, penalty=3., sigma=1.),
            2: dict(x=[0.3, 0.3], m=[2.], loss=0.4, alpha=0.05, penalty=2., sigma=1.),
            3: dict(x=[0.2, 0.2], m=[2.], loss=0., alpha=0.03, penalty=2., sigma=1.),
            4: dict(x=[0.1, 0.1], m=[2.], loss=0., alpha=0.01, penalty=1., sigma=1.),
        },
    ]

    visualizer = Visualizer(
        loss_fn=loss,
        eq_constr=[constr],
        solver_caches=fake_caches,
        cache_names=['temp1', 'temp2'],
        x1_bounds=(-2, 2),
        x2_bounds=(-2, 2),
        with_loss_plot=True,
        with_alpha_plot=True,
        with_penalty_plot=True,
    )

    visualizer.visualize()
