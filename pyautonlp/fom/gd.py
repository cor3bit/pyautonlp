import jax.numpy as jnp
from jax import grad, jit, vmap


def gd(
        obj_func,
        guess=None,
        learning_rate=None,
        max_iter=500,
        convergence=(10, 1e-6),
):
    # TODO check if guess is empty
    assert guess is not None
    x_curr = jnp.array(guess, dtype=jnp.float32)

    obj_func_dx = grad(obj_func)

    converged = False
    convergence_n, convergence_eps = convergence
    n_iter = 0
    n_small_diffs_in_a_row = 0
    while (not converged) and (n_iter < max_iter):
        direction = obj_func_dx(x_curr)
        x_next = x_curr - learning_rate * direction

        # check convergence
        diff = jnp.linalg.norm(x_next - x_curr, 2)
        if diff < convergence_eps:
            n_small_diffs_in_a_row += 1
        else:
            n_small_diffs_in_a_row = 0

        if n_small_diffs_in_a_row == convergence_n:
            converged = True

        x_curr = x_next
        n_iter += 1

    # fill additional info
    info = (converged, n_iter)

    return x_curr, info
