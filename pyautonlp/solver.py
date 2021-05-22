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
    def _get_log_str(n_iter, loss, step_size, conv_penalty):
        return f'Iteration {n_iter}: Loss {loss:.3f}, Alpha {step_size:.5f}, KKT violation {conv_penalty:.5f}.'
