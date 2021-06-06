from functools import partial
from typing import Tuple, Callable
from collections import namedtuple

import jax.numpy as jnp

from pyautonlp.constants import ConvergenceCriteria, LineSearch
from pyautonlp.solver import Solver

CacheItem = namedtuple('CacheItem', 'x m loss alpha x_dir H_pd penalty sigma')


class ConstrainedSolver(Solver):
    _eq_constr = None
    _ineq_constr = None
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

    def _eval_eq_constraints(self, x):
        if self._eq_constr is None:
            return .0

        return jnp.array([c_fn(x) for c_fn in self._eq_constr], dtype=jnp.float32)

    def _eval_ineq_constraints(self, x):
        if self._ineq_constr is None:
            return .0

        return jnp.array([c_fn(x) for c_fn in self._ineq_constr], dtype=jnp.float32)

    def _eval_constraints(self, x):
        return jnp.array([c_fn(x) for c_fn in self._constr_fns], dtype=jnp.float32)

    def _eval_constraints_with_slack(self, x, slack):
        eq_part = jnp.array([c_fn(x) for c_fn in self._eq_constr], dtype=jnp.float32)
        ineq_part = jnp.array([c_fn(x) + s for s, c_fn in zip(slack, self._ineq_constr)], dtype=jnp.float32)
        return jnp.concatenate((eq_part, ineq_part))

    def _eval_constraint_gradients(self, x):
        # should return size NxM matrix
        constraint_grads = jnp.empty((self._x_dims, self._multiplier_dims), dtype=jnp.float32)

        for j, c_grad_fn in enumerate(self._grad_constr_x_fns):
            constraint_grads = constraint_grads.at[:, j].set(c_grad_fn(x))

        return constraint_grads

    # def _grad_lagr_x_fn(self, x, multipliers):
    #     # should return size Nx1 vector
    #     return self._grad_loss_x_fn(x) + self._eval_constraint_gradients(x) @ multipliers

    def _null_space(self, A: jnp.ndarray):
        # mimics numpy null_space
        u, s, vh = jnp.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
        tol = jnp.amax(s) * rcond
        num = jnp.sum(s > tol, dtype=int)
        Q = vh[num:, :].T.conj()
        return Q

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

    def _kkt_violation(self, x_k, m_k, **kwargs):
        # TODO optimize
        max_c_eq = jnp.max(jnp.abs(self._eval_eq_constraints(x_k)))
        max_c_ineq = jnp.max(jnp.clip(self._eval_ineq_constraints(x_k), a_min=0))
        max_lagr_grad = jnp.max(jnp.abs(self._grad_lagr_x_fn(x_k, m_k)))

        max_viol = jnp.maximum(jnp.maximum(max_c_eq, max_c_ineq), max_lagr_grad)

        return max_viol <= self._tol, max_viol

    def _get_step_size_fn(
            self,
            strategy: str,
    ) -> Callable:
        if strategy == LineSearch.CONST:
            return self._constant_alpha
        elif strategy == LineSearch.BT:
            return partial(self._backtrack)
        elif strategy == LineSearch.BT_ARMIJO:
            return partial(self._backtrack, armijo=True)
        elif strategy == LineSearch.BT_MERIT:
            return partial(self._backtrack, merit=True)
        elif strategy == LineSearch.BT_MERIT_ARMIJO:
            return partial(self._backtrack, armijo=True, merit=True)
        else:
            raise ValueError(f'Unrecognized line search method: {strategy}.')

    def _constant_alpha(self, **kwargs):
        return self._alpha

    def _backtrack(self, x_k, grad_loss_x, direction, armijo=False, merit=False, max_iter=10):
        alpha = 1.

        direction_x = direction[:self._x_dims]

        loss_eval_fn = self._merit_fn if merit else self._loss_fn

        curr_loss = loss_eval_fn(x_k)
        next_loss = loss_eval_fn(x_k + alpha * direction_x)
        armijo_adj = self._calc_armijo_adj(x_k, alpha, grad_loss_x, direction_x, armijo, merit)

        n_iter = 0
        while (next_loss >= curr_loss + armijo_adj) and (n_iter < max_iter):
            alpha *= self._beta

            # update all alpha dependent
            next_loss = loss_eval_fn(x_k + alpha * direction_x)
            armijo_adj = self._calc_armijo_adj(x_k, alpha, grad_loss_x, direction_x, armijo, merit)

            n_iter += 1

        if n_iter == max_iter:
            alpha = .001
            self._logger.warning(f'fBacktracking failed to find alpha, using default value {alpha}.')

        return alpha

    def _merit_adj(self, x):
        eq_norm = 0. if self._eq_constr is None else jnp.linalg.norm(self._eval_eq_constraints(x), ord=1)

        ineq_norm = 0. if self._ineq_constr is None else jnp.linalg.norm(
            jnp.clip(self._eval_ineq_constraints(x), a_min=0), ord=1)

        return self._sigma * (eq_norm + ineq_norm)

    def _merit_fn(self, x):
        return self._loss_fn(x) + self._merit_adj(x)

    def _calc_armijo_adj(self, x, alpha, grad_loss_x, direction_x, armijo, merit):
        if not armijo:
            return 0.

        direct_deriv = jnp.dot(grad_loss_x, direction_x)
        if not merit:
            return self._gamma * alpha * direct_deriv

        return self._gamma * alpha * (direct_deriv - self._merit_adj(x))
