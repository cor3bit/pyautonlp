from typing import Callable

import matplotlib.pyplot as plt
import jax.numpy as jnp

from pyautonlp.constants import IntegrateMethod


def integrate(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.RK4,
) -> jnp.ndarray:
    if method == IntegrateMethod.EEULER:
        return _eeuler(fn, x0, u0, time_grid)
    elif method == IntegrateMethod.SSC_EEULER:
        return _eeuler_adaptive(fn, x0, u0, time_grid)
    elif method == IntegrateMethod.RK4:
        return _erk4(fn, x0, u0, time_grid)
    elif method == IntegrateMethod.IEULER:
        return _ieuler(fn, x0, u0, time_grid)
    else:
        raise ValueError(f'Unrecognized integration method: {method}.')


def _eeuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> jnp.ndarray:
    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # dx/dt - slope
        s_i = fn(x_i, u0, t_i)

        # new state
        x_j = x_i + h * s_i
        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res)


def _eeuler_adaptive(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        t_rel: float = 0.,
        t_abs: float = 1.,
        plot_h: bool = False,
) -> jnp.ndarray:
    res = [x0]
    x_i = x0
    t = time_grid[0]
    h = time_grid[1] - time_grid[0]
    tn = time_grid[-1]
    s_i = fn(x_i, u0, t)

    h_series = [h]

    while not jnp.isclose(t, tn):
        # correction for last time step
        if t + h > tn:
            h = tn - t

        x_j = x_i + h * s_i
        s_j = fn(x_j, u0, t + h)

        e = jnp.linalg.norm(s_j - s_i, ord=2) / 2
        # TODO t_rel, t_abs

        # make a step if error is small
        if e <= 1.:
            res.append(x_j)
            x_i = x_j
            s_i = s_j
            t += h

        # update interval h
        h *= jnp.minimum(2., jnp.maximum(.5, .9 / jnp.sqrt(e)))
        h_series.append(h)

    if plot_h:
        plt.plot(h_series)
        plt.show()

    return jnp.stack(res)


def _erk4(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> jnp.ndarray:
    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # kappas
        k1 = fn(x_i, u0, t_i)
        k2 = fn(x_i + h * k1 / 2, u0, t_i + h / 2)
        k3 = fn(x_i + h * k2 / 2, u0, t_i + h / 2)
        k4 = fn(x_i + h * k3, u0, t_i + h)

        # new state
        x_j = x_i + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res)


def _ieuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> jnp.ndarray:
    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # x'(t)
        s_i = fn(x_i, u0, t_i)

        # new state
        x_j = x_i + h * s_i
        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res)
