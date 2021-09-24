from typing import Tuple, Callable

import numpy as np
import jax.numpy as jnp
from quadprog import solve_qp

from pyautonlp.constants import ConvergenceCriteria, LineSearch
from pyautonlp.solver import Solver


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
    _armijo = False
    _merit = False

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    def _lagrangian(self, x, multipliers):
        return self._loss_fn(x) + jnp.sum(jnp.array([c_fn(x) for c_fn in self._constr_fns]) * multipliers)

    def _eval_eq_constraints(self, x, **kwargs):
        if self._eq_constr is None:
            return .0

        return jnp.array([c_fn(x) for c_fn in self._eq_constr], dtype=jnp.float32)

    def _eval_ineq_constraints(self, x, **kwargs):
        if self._ineq_constr is None:
            return .0

        return jnp.array([c_fn(x) for c_fn in self._ineq_constr], dtype=jnp.float32)

    def _eval_constraints(self, x, **kwargs):
        return jnp.array([c_fn(x) for c_fn in self._constr_fns], dtype=jnp.float32)

    def _eval_constraints_with_slack(self, x, slack):
        eq_part = jnp.array([c_fn(x) for c_fn in self._eq_constr], dtype=jnp.float32)
        ineq_part = jnp.array([c_fn(x) + s for s, c_fn in zip(slack, self._ineq_constr)], dtype=jnp.float32)
        return jnp.concatenate((eq_part, ineq_part))

    def _eval_eq_constraint_gradients(self, x):
        # should return size NxM_g matrix
        eq_constr_grads = jnp.empty((self._n_x, self._eq_mult_dims), dtype=jnp.float32)

        for j, c_grad_fn in enumerate(self._grad_constr_x_fns[:self._eq_mult_dims]):
            eq_constr_grads = eq_constr_grads.at[:, j].set(c_grad_fn(x))

        return eq_constr_grads

    def _eval_constraint_gradients(self, x):
        # should return size NxM matrix
        constraint_grads = jnp.empty((self._n_x, self._multiplier_dims), dtype=jnp.float32)

        for j, c_grad_fn in enumerate(self._grad_constr_x_fns):
            constraint_grads = constraint_grads.at[:, j].set(c_grad_fn(x))

        return constraint_grads

    # def _grad_lagr_x_fn(self, x, multipliers):
    #     # should return size Nx1 vector
    #     return self._grad_loss_x_fn(x) + self._eval_constraint_gradients(x) @ multipliers

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
            return self._backtrack
        elif strategy == LineSearch.BT_ARMIJO:
            self._armijo = True
            return self._backtrack
        elif strategy == LineSearch.BT_MERIT:
            self._merit = True
            return self._backtrack
        elif strategy == LineSearch.BT_MERIT_ARMIJO:
            self._merit = True
            self._armijo = True
            return self._backtrack
        else:
            raise ValueError(f'Unrecognized line search method: {strategy}.')

    def _constant_alpha(
            self,
            **kwargs
    ):
        return self._alpha

    def _backtrack(
            self,
            x_k,
            direction,
            grad_loss_x=None,
            initial_alpha=1.,
            max_iter=30,
            **kwargs
    ):
        alpha = initial_alpha
        direction_x = direction[:self._n_x]
        loss_eval_fn = self._merit_fn if self._merit else self._loss_fn

        curr_loss = loss_eval_fn(x_k)
        next_loss = loss_eval_fn(x_k + alpha * direction_x)
        armijo_adj = self._calc_armijo_adj(x_k, alpha, direction_x, grad_loss_x)

        n_iter = 0
        while (next_loss >= curr_loss + armijo_adj) and (n_iter < max_iter):
            alpha *= self._beta

            # update all alpha dependent
            next_loss = loss_eval_fn(x_k + alpha * direction_x)
            armijo_adj = self._calc_armijo_adj(x_k, alpha, direction_x, grad_loss_x)

            n_iter += 1

        if n_iter == max_iter:
            self._logger.warning(f'Backtracking failed to find alpha after {max_iter} iterations!')

        return alpha

    def _merit_adj(self, x):
        eq_norm = 0. if self._eq_constr is None else jnp.linalg.norm(self._eval_eq_constraints(x), ord=1)

        ineq_norm = 0. if self._ineq_constr is None else jnp.linalg.norm(
            jnp.clip(self._eval_ineq_constraints(x), a_min=0), ord=1)

        return self._sigma * (eq_norm + ineq_norm)

    def _merit_fn(self, x):
        return self._loss_fn(x) + self._merit_adj(x)

    def _calc_armijo_adj(self, x, alpha, direction_x, grad_loss_x):
        if not self._armijo:
            return 0.

        direct_deriv = jnp.dot(grad_loss_x, direction_x)
        if not self._merit:
            return self._gamma * alpha * direct_deriv

        return self._gamma * alpha * (direct_deriv - self._merit_adj(x))

    @staticmethod
    def _solve_qp(B_k, grad_loss_x, c_k, constr_grad_x, n_eq):
        # convert to numpy
        qd_G = np.array(B_k, dtype=np.double)
        qd_a = np.array(grad_loss_x, dtype=np.double)
        qd_b = np.array(c_k, dtype=np.double)
        qd_C = np.array(constr_grad_x, dtype=np.double)

        # solve QP
        qd_xf, qd_f, qd_xu, qd_iters, qd_lagr, qd_iact = solve_qp(
            qd_G, qd_a, qd_C, qd_b, meq=n_eq)  # , factorized=True

        # translate back to JAX
        d_k = jnp.array(np.concatenate((qd_xf, qd_lagr)))
        d_k *= -1.

        return d_k

    def _solve_infeasible_qp(
            self,
            B_k: jnp.ndarray,
            grad_loss_x: jnp.ndarray,
            c_eq_k: jnp.ndarray,
            c_ineq_k: jnp.ndarray,
            grad_c_eq_k: jnp.ndarray,
            grad_c_ineq_k: jnp.ndarray,
            rho: float = 10,
    ) -> jnp.ndarray:
        # modify matrices for soft constraints
        n_w = self._n_w
        n_g = self._n_eq_mult
        n_h = self._n_ineq_mult
        n_s = 2 * n_g + n_h

        new_B_k = jnp.block([
            [B_k, jnp.zeros((n_w, n_s))],
            [jnp.zeros((n_s, n_w)), rho * jnp.eye(n_s)],
        ])

        new_grad_loss_x = jnp.concatenate([grad_loss_x, jnp.zeros((n_s,))])

        new_c_k = jnp.concatenate([c_eq_k, c_ineq_k, jnp.zeros((n_s,))])

        n_g_prev = grad_c_eq_k.shape[0]
        n_h_prev = grad_c_ineq_k.shape[0]

        new_constr_grad_x = jnp.block([
            [grad_c_eq_k, -jnp.ones((n_g_prev, n_g)), jnp.ones((n_g_prev, n_g)), jnp.zeros((n_g_prev, n_h))],
            [grad_c_ineq_k, jnp.zeros((n_h_prev, 2 * n_g)), -jnp.ones((n_h_prev, n_h))],
            [jnp.zeros((n_s, n_w)), -jnp.eye(n_s)],
        ])

        # TODO remove numpy debug
        # np_new_B_k = np.array(new_B_k)
        # np_new_grad_loss_x = np.array(new_grad_loss_x)
        # np_new_c_k = np.array(new_c_k)
        # np_new_constr_grad_x = np.array(new_constr_grad_x)

        # solve QP
        d_k = self._solve_qp(
            new_B_k,
            new_grad_loss_x,
            new_c_k,
            new_constr_grad_x.T,
            self._n_eq_mult,
        )

        # re-assemble to remove slack var info
        d_k_w = d_k[:n_w]
        d_k_lagr = d_k[n_w + n_s: n_w + n_s + n_g_prev + n_h_prev]

        return jnp.concatenate([d_k_w, d_k_lagr])
