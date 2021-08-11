from typing import Callable, Tuple

import jax.numpy as jnp

from pyautonlp.constants import IntegrateMethod
from pyautonlp.root.newton_root import NewtonRoot


def integrate(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.RK4,
        **kwargs,
) -> jnp.ndarray:
    if method == IntegrateMethod.EEULER:
        return _eeuler(fn, x0, u0, time_grid)
    elif method == IntegrateMethod.SSC_EEULER:
        return _eeuler_adaptive(fn, x0, u0, time_grid, **kwargs)
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
) -> Tuple[jnp.ndarray, dict]:
    info = {}

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

    return jnp.stack(res), info


def _eeuler_adaptive(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        t_rel: float = 0.,
        t_abs: float = 1.,
) -> Tuple[jnp.ndarray, dict]:
    info = {}

    res = [x0]

    x_i = x0
    t = time_grid[0]
    h = time_grid[1] - time_grid[0]
    tn = time_grid[-1]
    s_i = fn(x_i, u0, t)

    ts = [t]

    while not jnp.isclose(t, tn):
        # correction for last time step
        if t + h > tn:
            h = tn - t

        x_j = x_i + h * s_i
        s_j = fn(x_j, u0, t + h)

        e = 0.5 * jnp.linalg.norm(s_j - s_i, ord=2) / t_abs
        # TODO t_rel

        # make a step if error is small
        if e <= 1.:
            res.append(x_j)
            ts.append(t + h)
            x_i = x_j
            s_i = s_j
            t += h

        # update interval h
        h *= jnp.minimum(2., jnp.maximum(.5, .9 / jnp.sqrt(e)))

    info['t'] = ts

    return jnp.stack(res), info


def _erk4(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    info = {}

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

    return jnp.stack(res), info


def _ieuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    info = {}

    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # x'(t)
        s_i = fn(x_i, u0, t_i)

        # new state
        guess_x_j = x_i + h * s_i

        solver = NewtonRoot(
            lambda x: x_i + h * fn(x, u0, t_j) - x,
            guess_x_j,
            tol=1e-7,
            verbose=False,
        )
        x_j = solver.solve()

        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res), info
