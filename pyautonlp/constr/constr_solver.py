from functools import partial
from typing import Tuple, Callable

import jax.numpy as jnp

from pyautonlp.constants import ConvergenceCriteria, LearningRateStrategy
from pyautonlp.solver import Solver


class ConstrainedSolver(Solver):
    _constr_fns = None
    _grad_loss_x_fn = None
    _grad_constr_x_fns = None
    _multiplier_dims = None
    _alpha = None
    _beta = None
    _gamma = None
    _sigma = None
    _tol = None

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    def _lagrangian(self, x, multipliers):
        return self._loss_fn(x) + jnp.sum(jnp.array([c_fn(x) for c_fn in self._constr_fns]) * multipliers)

    def _eval_constraints(self, x):
        return jnp.array([c_fn(x) for c_fn in self._constr_fns], dtype=jnp.float32)

    def _eval_constraint_gradients(self, x):
        # should return size NxM matrix
        constraint_grads = jnp.empty((self._x_dims, self._multiplier_dims), dtype=jnp.float32)

        for j, c_grad_fn in enumerate(self._grad_constr_x_fns):
            constraint_grads = constraint_grads.at[:, j].set(c_grad_fn(x))

        return constraint_grads

    def _grad_lagr_x_fn(self, x, multipliers):
        # should return size Nx1 vector
        return self._grad_loss_x_fn(x) + self._eval_constraint_gradients(x) @ multipliers

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

    def _kkt_violation(self, curr_x, curr_m, **kwargs):
        max_c_violation = jnp.max(jnp.abs(self._eval_constraints(curr_x)))
        max_lagr_grad_violation = jnp.max(jnp.abs(self._grad_lagr_x_fn(curr_x, curr_m)))
        max_viol = jnp.maximum(max_c_violation, max_lagr_grad_violation)
        return max_viol <= self._tol, max_viol

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
        elif strategy == LearningRateStrategy.BT_MERIT:
            return partial(self._backtrack, merit=True)
        elif strategy == LearningRateStrategy.BT_MERIT_ARMIJO:
            return partial(self._backtrack, armijo=True, merit=True)
        else:
            raise ValueError(f'Unrecognized learning rate strategy: {strategy}.')

    def _constant_alpha(self, **kwargs):
        return self._alpha

    def _backtrack(self, curr_x, grad_loss_x, direction, armijo=False, merit=False, max_iter=7):
        # TODO check that starts at 1 (or more?) and not uses prev value
        alpha = 1.

        direction_x = direction[:self._x_dims]

        loss_eval_fn = self._merit_fn if merit else self._loss_fn

        curr_loss = loss_eval_fn(curr_x)
        next_loss = loss_eval_fn(curr_x + alpha * direction_x)
        armijo_adj = self._calc_armijo_adj(curr_x, alpha, grad_loss_x, direction_x, armijo, merit)

        n_iter = 0
        while (next_loss >= curr_loss + armijo_adj) and (n_iter < max_iter):
            alpha *= self._beta

            # update all alpha dependent
            next_loss = loss_eval_fn(curr_x + alpha * direction_x)
            armijo_adj = self._calc_armijo_adj(curr_x, alpha, grad_loss_x, direction_x, armijo, merit)

            n_iter += 1

        return alpha

    def _merit_fn(self, x):
        return self._loss_fn(x) + self._sigma * jnp.linalg.norm(self._eval_constraints(x), ord=1)

    def _calc_armijo_adj(self, curr_x, alpha, grad_loss_x, direction_x, armijo, merit):
        if not armijo:
            return 0.

        direct_deriv = jnp.dot(grad_loss_x, direction_x)
        if not merit:
            return self._gamma * alpha * direct_deriv

        return self._gamma * alpha * (
                    direct_deriv - self._sigma * jnp.linalg.norm(self._eval_constraints(curr_x), ord=1))
