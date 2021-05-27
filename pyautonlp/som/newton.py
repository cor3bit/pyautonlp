import logging
from typing import List, Tuple, Callable

import jax.numpy as jnp
from jax import grad, jit, vmap

from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch
# from pyautonlp.utils import hessian


def newton(
        loss_fn: Callable,
        guess: jnp.ndarray = None,
        hessian_approx: str = Direction.EXACT_NEWTON,
        lr_strategy: str = LineSearch.CONST,
        lr: float = 0.01,
        alpha: float = None,  # relevant if lr strategy is Backtracking
        beta: float = None,  # relevant if lr strategy is Backtracking
        max_iter: int = 500,
        conv_criteria: str = ConvergenceCriteria.STEP_DIFF_NORM,
        conv_params: Tuple = (10, 1e-6),
        verbose: bool = False,
        visualize: bool = False,
) -> Tuple[jnp.ndarray, Tuple]:
    #
    logger = logging.getLogger('newton_solver')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # logger.info(f'Running ')

    # TODO check if guess is empty
    assert guess is not None
    x_curr = jnp.array(guess, dtype=jnp.float32)

    # precompile Jacobian & Hessian functions
    grad_fn = grad(loss_fn)
    hess_fn = hessian(loss_fn)

    # run iteration loop
    converged = False
    convergence_n, convergence_eps = conv_params
    n_iter = 0
    n_small_diffs_in_a_row = 0
    while (not converged) and (n_iter < max_iter):
        grad_at_x = grad_fn(x_curr)
        hess_at_x = hess_fn(x_curr)
        direction = jnp.linalg.solve(hess_at_x, grad_at_x)
        x_next = x_curr - lr * direction

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
