from typing import Callable, Tuple, Dict

import jax.numpy as jnp
from jax import jacfwd

from pyautonlp.constants import IntegrateMethod
from pyautonlp.root.newton_root import NewtonRoot


def integrate(
        fn: Callable,
        x0: jnp.ndarray,
        u: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.RK4,
        **kwargs,
) -> Tuple[jnp.ndarray, Dict]:
    if method == IntegrateMethod.EEULER:
        return _eeuler(fn, x0, u, time_grid)
    elif method == IntegrateMethod.SSC_EEULER:
        return _eeuler_adaptive(fn, x0, u, time_grid, **kwargs)
    elif method == IntegrateMethod.RK4:
        return _erk4(fn, x0, u, time_grid)
    elif method == IntegrateMethod.IEULER:
        return _ieuler(fn, x0, u, time_grid)
    else:
        raise ValueError(f'Unrecognized integration method: {method}.')


def integrate_with_end(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.RK4,
        eps: float = 1e-5,
        **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # base
    xs, info = integrate(fn, x0, u0, time_grid, method=method, **kwargs)
    x = xs[-1]

    # Calculate G_x
    x_dims = x.shape[0]
    G_x = []

    for i in range(x_dims):
        delta_i = jnp.zeros_like(x).at[i].set(eps)
        x0_ = x0 + delta_i
        xs_, _ = integrate(fn, x0_, u0, time_grid, method=method, **kwargs)
        x_ = xs_[-1]
        g_x_i = (x_ - x) / eps
        G_x.append(g_x_i)

    G_x = jnp.stack(G_x).T

    # TODO G_u
    G_u = None

    return x, G_x, G_u


def integrate_with_ind(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.SSC_EEULER,
        eps: float = 1e-5,
        **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if method != IntegrateMethod.SSC_EEULER:
        raise ValueError(f'Method {method} supported for IND.')

    # base
    xs, info = integrate(fn, x0, u0, time_grid, method=method, **kwargs)
    time_grid_base = info['ts']
    x = xs[-1]

    # Calculate G_x
    x_dims = x.shape[0]
    G_x = []

    for i in range(x_dims):
        delta_i = jnp.zeros_like(x).at[i].set(eps)
        x0_ = x0 + delta_i
        xs_, _ = integrate(fn, x0_, u0, time_grid_base, method=IntegrateMethod.EEULER, **kwargs)
        x_ = xs_[-1]
        g_x_i = (x_ - x) / eps
        G_x.append(g_x_i)

    G_x = jnp.stack(G_x).T

    # TODO G_u
    G_u = None

    return x, G_x, G_u


def integrate_with_ad(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.RK4,
        **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def _integrate_last(x0_inner, u0_inner, time_grid_inner, method_inner, fn_inner, **kwargs_inner):
        xs_inner, _ = integrate(fn_inner, x0_inner, u0_inner, time_grid_inner, method=method_inner, **kwargs_inner)
        return xs_inner[-1]

    xs, _ = integrate(fn, x0, u0, time_grid, method=method, **kwargs)
    x = xs[-1]
    G_x = jacfwd(_integrate_last)(x0, u0, time_grid, method, fn, **kwargs)
    G_u = None

    return x, G_x, G_u


def ad_eeuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        **kwargs,
):
    I = jnp.eye(2)
    A = jnp.eye(2)

    h = time_grid[1] - time_grid[0]

    dfds = jnp.array([
        [-16., 12.],
        [12., -9.],
    ], dtype=jnp.float64)

    for i in range(len(time_grid) - 1):
        delta_A = I + h * dfds
        A = delta_A @ A

    G_x = A.T

    return G_x


def ad_erk4(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        **kwargs,
):
    def rk_stage(dynamics, t, x, u, A):
        dfds = jnp.array([
            [-16., 12.],
            [12., -9.],
        ], dtype=jnp.float64)

        k = dynamics(x, u, t)
        k_a = dfds @ A
        return k, k_a

    s = x0
    I = jnp.eye(2)
    A = jnp.eye(2)

    for i in range(len(time_grid) - 1):
        t_k = time_grid[i]
        t = time_grid[i + 1]
        h = time_grid[i + 1] - time_grid[i]

        k_1, k_a_1 = rk_stage(fn, t, s, u0, I)
        k_2, k_a_2 = rk_stage(fn, t + h / 2, s + h / 2 * k_1, u0, I + h / 2 * k_a_1)
        k_3, k_a_3 = rk_stage(fn, t + h / 2, s + h / 2 * k_2, u0, I + h / 2 * k_a_2)
        k_4, k_a_4 = rk_stage(fn, t + h, s + h * k_3, u0, I + h * k_a_3)

        s = s + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        M = I + h / 6 * (k_a_1 + 2 * k_a_2 + 2 * k_a_3 + k_a_4)
        A = M @ A

    G_x = A.T

    return G_x


def ad_ieuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        **kwargs,
):
    I = jnp.eye(2)
    A = jnp.eye(2)

    # h = time_grid[1] - time_grid[0]

    dfds = jnp.array([
        [-16., 12.],
        [12., -9.],
    ], dtype=jnp.float64)

    G_x = None
    G_u = None

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
            record_ad=True,
        )
        x_j, solver_info = solver.solve()

        # AD sensitivity
        jac_r_k = solver_info['ad']
        drdk_inv = jnp.linalg.inv(jac_r_k)
        delta_A = I - h * drdk_inv @ dfds
        A = delta_A @ A

        res.append(x_j)

        # update vars
        x_i = x_j

    G_x = A.T

    return G_x


def _eeuler(
        fn: Callable,
        x0: jnp.ndarray,
        u: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    is_const_u = len(u.shape) == 1
    if not is_const_u:
        # Make sure u(t) exists for all times except for the last time step
        assert len(time_grid) - 1 == u.shape[0]

    res = [x0]
    x_i = x0
    for i, (t_i, t_j) in enumerate(zip(time_grid[0:-1], time_grid[1:])):
        h = t_j - t_i

        u_i = u if is_const_u else u[i]

        # dx/dt - slope
        s_i = fn(x_i, u_i, t_i)

        # new state
        x_j = x_i + h * s_i
        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res), info


def _eeuler_adaptive(
        fn: Callable,
        x0: jnp.ndarray,
        u: jnp.ndarray,
        time_grid: jnp.ndarray,
        t_rel: float = 0.,
        t_abs: float = 1.,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    res = [x0]

    is_const_u = len(u.shape) == 1
    if not is_const_u:
        raise ValueError(f'Non-constant u not supported for {IntegrateMethod.SSC_EEULER}.')

    x_i = x0
    t = time_grid[0]
    h = time_grid[1] - time_grid[0]
    tn = time_grid[-1]
    s_i = fn(x_i, u, t)

    ts = [t]
    hs = []

    while not jnp.isclose(t, tn):
        # correction for last time step
        if t + h > tn:
            h = tn - t

        x_j = x_i + h * s_i
        s_j = fn(x_j, u, t + h)

        e = 0.5 * jnp.linalg.norm(s_j - s_i, ord=2) / t_abs
        # TODO t_rel

        # make a step if error is small
        if e <= 1.:
            res.append(x_j)
            ts.append(t + h)
            hs.append(h)
            x_i = x_j
            s_i = s_j
            t += h

        # update interval h
        h *= jnp.minimum(2., jnp.maximum(.5, .9 / jnp.sqrt(e)))

    info['ts'] = ts
    info['hs'] = hs

    return jnp.stack(res), info


def _erk4(
        fn: Callable,
        x0: jnp.ndarray,
        u: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    is_const_u = len(u.shape) == 1
    if not is_const_u:
        # Make sure u(t) exists for all times except for the last time step
        assert len(time_grid) - 1 == u.shape[0]

    res = [x0]
    x_i = x0
    for i, (t_i, t_j) in enumerate(zip(time_grid[0:-1], time_grid[1:])):
        h = t_j - t_i

        u_i = u if is_const_u else u[i]

        # kappas
        k1 = fn(x_i, u_i, t_i)
        k2 = fn(x_i + h * k1 / 2, u_i, t_i + h / 2)
        k3 = fn(x_i + h * k2 / 2, u_i, t_i + h / 2)
        k4 = fn(x_i + h * k3, u_i, t_i + h)

        # new state
        x_j = x_i + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res), info


def _ieuler(
        fn: Callable,
        x0: jnp.ndarray,
        u: jnp.ndarray,
        time_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    is_const_u = len(u.shape) == 1
    if not is_const_u:
        # Make sure u(t) exists for all times except for the last time step
        assert len(time_grid) - 1 == u.shape[0]

    res = [x0]
    x_i = x0
    for i, (t_i, t_j) in enumerate(zip(time_grid[0:-1], time_grid[1:])):
        h = t_j - t_i

        u_i = u if is_const_u else u[i]

        # x'(t)
        s_i = fn(x_i, u_i, t_i)

        # new state
        guess_x_j = x_i + h * s_i

        solver = NewtonRoot(
            lambda x: x_i + h * fn(x, u_i, t_j) - x,
            guess_x_j,
            tol=1e-7,
            verbose=False,
        )
        x_j, _ = solver.solve()

        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res), info
