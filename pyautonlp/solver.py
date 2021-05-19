from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp

from pyautonlp.constants import ConvergenceCriteria
from pyautonlp.convergence import kkt_violation


class Solver(ABC):
    _loss_fn = None

    @abstractmethod
    def solve(
            self,
    ) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    def _get_convergence_fn(
            self,
            criteria: str,
    ) -> Callable:
        if criteria == ConvergenceCriteria.KKT_VIOLATION:
            return kkt_violation
        elif criteria == ConvergenceCriteria.STEP_DIFF_NORM:
            raise NotImplementedError
        else:
            raise ValueError(f'Unrecognized convergence criteria: {criteria}.')
