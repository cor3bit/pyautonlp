from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp


class Solver(ABC):
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
