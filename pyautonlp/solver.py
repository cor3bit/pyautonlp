from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp


class Solver(ABC):
    _loss_fn = None
    _x_dims = None
    _cache = None  # save solver info for charting

    @abstractmethod
    def solve(
            self,
    ) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    @staticmethod
    def _get_log_str(k, cache_item):
        return f'Iteration {k}: Loss {cache_item.loss:.3f}, ' \
               f'Alpha {cache_item.alpha:.5f}, Sigma {cache_item.sigma:.4f}, ' \
               f'KKT violation {cache_item.penalty:.5f}.'

    @staticmethod
    def _is_pd_matrix(a: jnp.ndarray) -> bool:
        try:
            L = jnp.linalg.cholesky(a)
            if jnp.isnan(L).any():
                return False
            return True
        except Exception as e:
            return False
