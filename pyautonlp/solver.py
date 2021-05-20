from abc import ABC, abstractmethod
from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp

from pyautonlp.constants import ConvergenceCriteria, LearningRateStrategy


class Solver(ABC):
    _loss_fn = None
    _x_dims = None
    _alpha = None
    _beta = None
    _tol = None
    _cache = None  # save solver info for charting

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
            return self._kkt_violation
        elif criteria == ConvergenceCriteria.STEP_DIFF_NORM:
            raise NotImplementedError
        else:
            raise ValueError(f'Unrecognized convergence criteria: {criteria}.')

    def _kkt_violation(self, grad_at_point, **kwargs):
        return jnp.max(jnp.abs(grad_at_point)) <= self._tol

    def _get_step_size_fn(
            self,
            strategy: str,
    ) -> Callable:
        if strategy == LearningRateStrategy.CONST:
            return self._constant_alpha
        elif strategy == LearningRateStrategy.BT:
            return partial(self._backtrack)
        elif strategy == LearningRateStrategy.BT_ARMIJO:
            return partial(self._backtrack, armijo=True)
        elif strategy == LearningRateStrategy.BT_WOLFE:
            raise NotImplementedError
        else:
            raise ValueError(f'Unrecognized learning rate strategy: {strategy}.')

    def _constant_alpha(self, **kwargs):
        return self._alpha

    def _backtrack(self, curr_state, direction, armijo=False, wolfe=False, max_iter=10, **kwargs):
        # TODO check that starts at 1 (or more?) and not uses prev value
        alpha = 1.0

        curr_x = curr_state[:self._x_dims]
        direction_x = direction[:self._x_dims]
        curr_loss = self._loss_fn(curr_x)

        next_loss = self._loss_fn(curr_x - alpha * direction_x)

        # if armijo:
        #     curr_loss += self.gamma * alpha * jnp.dot(grattt, )

        n_iter = 0
        while next_loss >= curr_loss and n_iter < max_iter:
            alpha *= self._beta
            next_loss = self._loss_fn(curr_x - alpha * direction_x)
            n_iter += 1

        return alpha
