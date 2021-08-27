from typing import Callable, Tuple, Dict

import jax.numpy as jnp
from jax import jacfwd

from pyautonlp.constants import IntegrateMethod
from pyautonlp.root.newton_root import NewtonRoot


def integrate(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        method: str = IntegrateMethod.EEULER,
        with_grads: bool = False,
        dfds_fn: Callable = None,
        dfdu_fn: Callable = None,
        **kwargs,
) -> Tuple[jnp.ndarray, Dict]:
    if method == IntegrateMethod.EEULER:
        return _eeuler(fn, x0, u0, time_grid, with_grads, dfds_fn, dfdu_fn)
    elif method == IntegrateMethod.SSC_EEULER:
        return _eeuler_adaptive(fn, x0, u0, time_grid, with_grads, dfds_fn, dfdu_fn, **kwargs)
    elif method == IntegrateMethod.RK4:
        return _erk4(fn, x0, u0, time_grid, with_grads, dfds_fn, dfdu_fn)
    elif method == IntegrateMethod.IEULER:
        return _ieuler(fn, x0, u0, time_grid, with_grads, dfds_fn, dfdu_fn)
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
    G_x = []
    x_dims = x.shape[0]
    for i in range(x_dims):
        delta_i = jnp.zeros_like(x).at[i].set(eps)
        x0_ = x0 + delta_i
        xs_, _ = integrate(fn, x0_, u0, time_grid, method=method, **kwargs)
        x_ = xs_[-1]
        g_x_i = (x_ - x) / eps
        G_x.append(g_x_i)

    G_x = jnp.stack(G_x).T

    # Calculate G_u
    G_u = []
    u_dims = u0.shape[0]
    for j in range(u_dims):
        delta_j = jnp.zeros_like(u0).at[j].set(eps)
        u0_ = u0 + delta_j
        xs_, _ = integrate(fn, x0, u0_, time_grid, method=method, **kwargs)
        x_ = xs_[-1]
        g_u_j = (x_ - x) / eps
        G_u.append(g_u_j)

    G_u = jnp.stack(G_u).T

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
        raise ValueError(f'Method {method} not supported for IND.')

    # base
    xs, info = integrate(fn, x0, u0, time_grid, method=method, **kwargs)
    time_grid_base = info['ts']
    x = xs[-1]

    # Calculate G_x
    G_x = []
    x_dims = x.shape[0]
    for i in range(x_dims):
        delta_i = jnp.zeros_like(x).at[i].set(eps)
        x0_ = x0 + delta_i
        xs_, _ = integrate(fn, x0_, u0, time_grid_base, method=IntegrateMethod.EEULER, **kwargs)
        x_ = xs_[-1]
        g_x_i = (x_ - x) / eps
        G_x.append(g_x_i)

    G_x = jnp.stack(G_x).T

    # Calculate G_u
    G_u = []
    u_dims = u0.shape[0]
    for j in range(u_dims):
        delta_j = jnp.zeros_like(u0).at[j].set(eps)
        u0_ = u0 + delta_j
        xs_, _ = integrate(fn, x0, u0_, time_grid_base, method=IntegrateMethod.EEULER, **kwargs)
        x_ = xs_[-1]
        g_u_j = (x_ - x) / eps
        G_u.append(g_u_j)

    G_u = jnp.stack(G_u).T

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
    G_x, G_u = jacfwd(_integrate_last, (0, 1))(x0, u0, time_grid, method, fn, **kwargs)

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

    dfds = jnp.array([
        [-16., 12.],
        [12., -9.],
    ], dtype=jnp.float64)

    for i in range(len(time_grid) - 1):
        h = time_grid[i + 1] - time_grid[i]
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
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        with_grads: bool,
        dfds_fn: Callable = None,
        dfdu_fn: Callable = None,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    # sensitivities
    I, A, B = None, None, None
    if with_grads:
        assert dfds_fn is not None
        assert dfdu_fn is not None
        x_dims = x0.shape[0]
        u_dims = u0.shape[0]
        I = jnp.eye(x_dims)
        A = jnp.eye(x_dims)
        B = jnp.zeros((x_dims, u_dims))

    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # sensitivity
        if with_grads:
            # TODO check that indeed t_j, not t_i
            inc = I + h * dfds_fn(x_i, u0, t_j)
            A = inc @ A
            B = inc @ B + h * dfdu_fn(x_i, u0, t_j)

        # dx/dt - slope
        s_i = fn(x_i, u0, t_i)

        # new state
        x_j = x_i + h * s_i
        res.append(x_j)

        # update vars
        x_i = x_j

    # info
    if with_grads:
        info['G_x'] = A.T
        info['G_u'] = B.T

    return jnp.stack(res), info


def _eeuler_adaptive(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        with_grads: bool,
        dfds_fn: Callable = None,
        dfdu_fn: Callable = None,
        t_abs: float = 1.,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    res = [x0]

    x_i = x0
    t = time_grid[0]
    h = time_grid[1] - time_grid[0]
    tn = time_grid[-1]
    s_i = fn(x_i, u0, t)

    ts = [t]
    hs = []

    if with_grads:
        raise NotImplementedError

    while not jnp.isclose(t, tn):
        # correction for last time step
        if t + h > tn:
            h = tn - t

        x_j = x_i + h * s_i
        s_j = fn(x_j, u0, t + h)

        e = 0.5 * jnp.linalg.norm(s_j - s_i, ord=2) / t_abs

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

    # info
    info['ts'] = ts
    info['hs'] = hs

    return jnp.stack(res), info


def _rk_stage(dynamics, dfds_fn, dfdu_fn, x, u, t, A, B):
    k = dynamics(x, u, t)
    inc = dfds_fn(x, u, t)
    k_a = inc @ A
    k_b = inc @ B + dfdu_fn(x, u, t)
    return k, k_a, k_b


def _erk4(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        with_grads: bool,
        dfds_fn: Callable = None,
        dfdu_fn: Callable = None,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    # sensitivities
    I, A, Z, B = None, None, None, None
    if with_grads:
        assert dfds_fn is not None
        assert dfdu_fn is not None
        x_dims = x0.shape[0]
        u_dims = u0.shape[0]
        I = jnp.eye(x_dims)
        A = jnp.eye(x_dims)
        Z = jnp.zeros((x_dims, u_dims))
        B = jnp.zeros((x_dims, u_dims))

    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # sensitivity
        if with_grads:
            k1, k_a1, k_b1 = _rk_stage(fn, dfds_fn, dfdu_fn,
                                       x_i, u0, t_i, I, Z)
            k2, k_a2, k_b2 = _rk_stage(fn, dfds_fn, dfdu_fn,
                                       x_i + h / 2 * k1, u0, t_i + h / 2, I + h / 2 * k_a1, h / 2 * k_b1)
            k3, k_a3, k_b3 = _rk_stage(fn, dfds_fn, dfdu_fn,
                                       x_i + h / 2 * k2, u0, t_i + h / 2, I + h / 2 * k_a2, h / 2 * k_b2)
            k4, k_a4, k_b4 = _rk_stage(fn, dfds_fn, dfdu_fn,
                                       x_i + h * k3, u0, t_i + h, I + h * k_a3, h * k_b3)

            # new state
            x_j = x_i + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            M = I + h / 6 * (k_a1 + 2 * k_a2 + 2 * k_a3 + k_a4)
            A = M @ A
            B = M @ B + h / 6 * (k_b1 + 2 * k_b2 + 2 * k_b3 + k_b4)
        else:
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

    # info
    if with_grads:
        info['G_x'] = A
        info['G_u'] = B

    return jnp.stack(res), info


def _ieuler(
        fn: Callable,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        time_grid: jnp.ndarray,
        with_grads: bool,
        dfds_fn: Callable = None,
        dfdu_fn: Callable = None,
) -> Tuple[jnp.ndarray, Dict]:
    info = {}

    # sensitivities
    I, A, B = None, None, None
    if with_grads:
        assert dfds_fn is not None
        assert dfdu_fn is not None
        x_dims = x0.shape[0]
        u_dims = u0.shape[0]
        I = jnp.eye(x_dims)
        A = jnp.eye(x_dims)
        B = jnp.zeros((x_dims, u_dims))

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
            record_ad=with_grads,
        )
        x_j, solver_info = solver.solve()

        if with_grads:
            jac_r_k = solver_info['ad']
            drdk_inv = jnp.linalg.inv(jac_r_k)
            # TODO check that indeed t_j, not t_i
            inc = I - h * drdk_inv @ dfds_fn(x_i, u0, t_j)
            A = inc @ A
            B = inc @ B - h * drdk_inv @ dfdu_fn(x_i, u0, t_j)

        res.append(x_j)

        # update vars
        x_i = x_j

    # info
    if with_grads:
        info['G_x'] = A.T
        info['G_u'] = B.T

    return jnp.stack(res), info
