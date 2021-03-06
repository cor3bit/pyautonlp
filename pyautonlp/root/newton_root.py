import logging
from typing import Tuple, Callable, Union, Dict

import jax.numpy as jnp
from jax import jacfwd


class NewtonRoot:
    def __init__(
            self,
            residual_fn: Callable,
            guess: Union[Tuple, jnp.ndarray] = None,
            max_iter: int = 100,
            tol: float = 1e-6,
            verbose: bool = False,
            record_ad: bool = False,
    ):
        # logger
        self._logger = logging.getLogger('root_finder')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # functions
        self._residual_fn = residual_fn
        self._jac_residual_fn = jacfwd(residual_fn)

        # initial guess
        assert guess is not None
        self._x_dims = len(guess)
        self._initial_x = jnp.array(guess)
        # self._initial_x = jnp.ones(self._x_dims, dtype=jnp.float32)
        self._logger.info(f'Dimensions of the state vector: {self._x_dims}.')

        # convergence
        self._max_iter = max_iter
        self._tol = tol

        # AD sensitivity
        self._ad = record_ad

    def solve(self) -> Tuple[jnp.ndarray, Dict]:
        info = {}


        x_curr = self._initial_x
        self._logger.info(f'Initial guess is: {x_curr}.')

        r_k = self._residual_fn(x_curr)

        k = 0

        jac_r_k = None
        converged = False
        while (not converged) and (k < self._max_iter):
            # logging
            # self._logger.info(f'Iteration {k}: Residual {np.max(r_k):.6f}.')

            jac_r_k = self._jac_residual_fn(x_curr)
            inc = jnp.linalg.solve(jac_r_k, r_k)
            x_next = x_curr - inc

            # update variables
            x_curr = x_next
            r_k = self._residual_fn(x_curr)
            k += 1

            # check convergence
            converged = self._converged(r_k)

        if self._ad:
            assert jac_r_k is not None
            info['ad'] = jac_r_k

        # logging
        # self._logger.info(f'Iteration {k}: Residual {np.max(r_k):.6f}.')

        return x_curr, info

    def _converged(self, r_k: jnp.ndarray) -> bool:
        return jnp.allclose(r_k, jnp.zeros(self._x_dims), atol=self._tol)
