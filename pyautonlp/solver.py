from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd, jacrev


class Solver(ABC):
    _logger = None
    _loss_fn = None
    _x_dims = None
    _cache = None  # info collected during the run

    @abstractmethod
    def solve(
            self,
    ) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    @staticmethod
    def _get_log_str(k, cache_item):
        is_pd = '-' if cache_item.H_pd is None else str(cache_item.H_pd)

        return f'Iteration {k}: Loss: {cache_item.loss:.3f}; ' \
               f'Alpha: {cache_item.alpha:.6f}; ' \
               f'H is pd: {is_pd}; ' \
               f'Sigma: {cache_item.sigma:.2f}; ' \
               f'KKT Penalty: {cache_item.penalty:.5f}.'

    @staticmethod
    def _is_pd_matrix(a: jnp.ndarray) -> bool:
        try:
            L = jnp.linalg.cholesky(a)
            if jnp.isnan(L).any():
                return False
            return True
        except Exception as e:
            return False

    def _hessian_exact(self, fn: Callable) -> Callable:
        return jit(jacfwd(jacrev(fn)))

    def _hessian_bfgs_approx(self, B_prev, g_k, g_prev, x_k, x_prev, **kwargs) -> jnp.ndarray:
        if B_prev is None:
            return jnp.eye(N=self._x_dims)  # B_0
        else:
            y = x_k - x_prev
            s = g_k - g_prev

            c1 = jnp.inner(y, s)
            incr1 = jnp.outer(y, y) / c1

            c2 = s @ B_prev @ s
            incr2 = B_prev @ jnp.outer(s, s) @ B_prev / c2

            # TODO check BFGS
            if jnp.isclose(c1, 0) or jnp.isclose(c2, 0):
                self._logger.warning(f'BFGS: constants are zero.')
                return B_prev

            B_k = B_prev + incr1 - incr2

            return B_k

    def _hessian_sd_approx(self, **kwargs) -> jnp.ndarray:
        return jnp.eye(N=self._x_dims)
