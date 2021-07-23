import jax.numpy as jnp


def eeuler(fn, x0, u0, time_grid, adaptive=False) -> jnp.ndarray:
    res = [x0]
    x_i = x0
    for t_i, t_j in zip(time_grid[0:-1], time_grid[1:]):
        h = t_j - t_i

        # x'(t)
        s_i = fn(x_i, u0, t_i)

        # new state
        if not adaptive:
            x_j = x_i + h * s_i
        else:
            x_j = x_i + h * s_i

        res.append(x_j)

        # update vars
        x_i = x_j

    return jnp.stack(res)


def erk4(fn, x0, u0, time_grid) -> jnp.ndarray:
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


def ieuler(fn, x0, u0, time_grid) -> jnp.ndarray:
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

    # return jnp.array(res, dtype=jnp.float32)
    return jnp.stack(res)
