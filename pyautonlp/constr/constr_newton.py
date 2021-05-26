import logging
from typing import List, Tuple, Callable
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from jax import grad
import matplotlib.pyplot as plt
import seaborn as sns

from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization
from pyautonlp.utils import hessian
from pyautonlp.constr.constr_solver import ConstrainedSolver

CacheItem = namedtuple('CacheItem', 'x m loss alpha x_dir H_pd penalty sigma')


class ConstrainedNewtonSolver(ConstrainedSolver):
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            guess: jnp.ndarray = None,
            direction: str = Direction.STEEPEST_DESCENT,
            reg: str = HessianRegularization.NONE,
            line_search: str = LineSearch.CONST,
            alpha: float = 0.01,  # relevant for Constant step line search
            beta: float = 0.5,  # relevant for Backtracking line search
            gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            sigma: float = 1.0,  # relevant for Backtracking + Merit Function line search
            conv_criteria: str = ConvergenceCriteria.KKT_VIOLATION,
            conv_tol: float = 1e-8,
            max_iter: int = 500,
            verbose: bool = False,
            **kwargs
    ):
        # logger
        self._logger = logging.getLogger('newton_solver')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # loss
        self._loss_fn = loss_fn

        # initial guess
        # TODO fix guess is empty
        # TODO move guess to solve()
        assert guess is not None
        self._x_dims = len(guess)
        self._initial_x = jnp.array(guess, dtype=jnp.float32)

        self._logger.info(f'Dimensions of the state vector: {self._x_dims}.')

        # constraints
        if eq_constr is None or not eq_constr:
            raise ValueError('Equality constraints not found.')
        if ineq_constr is not None:
            raise ValueError('Inequality constraints found for Newton solver.')

        self._eq_constr = eq_constr
        self._constr_fns = eq_constr
        self._multiplier_dims = len(eq_constr)

        self._logger.info(f'Dimensions of the multiplier vector: {self._multiplier_dims}.')

        self._initial_multipliers = jnp.zeros(shape=(self._multiplier_dims,), dtype=jnp.float32)

        # convergence
        self._convergence_fn = self._get_convergence_fn(conv_criteria)
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        self._step_size_fn = self._get_step_size_fn(line_search)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma

        # save intermediate step info
        self._cache = {}

        # grad & hessian functions
        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1
        self._grad_constr_x_fns = [grad(f) for f in self._constr_fns]

        self._direction = direction
        if direction == Direction.EXACT_NEWTON:
            self._hess_lagr_xx_fn = hessian(self._lagrangian)  # N-by-N

        self._reg = reg

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        x_k = self._initial_x
        self._logger.info(f'Initial guess is: {x_k}.')

        m_k = self._initial_multipliers
        self._logger.info(f'Initial multipliers are: {m_k}.')

        kkt_n = self._x_dims + self._multiplier_dims
        kkt_matrix = jnp.zeros(shape=(kkt_n, kkt_n), dtype=jnp.float32)

        converged, conv_penalty = self._convergence_fn(x_k, m_k)
        k = 0
        while (not converged) and (k < self._max_iter):
            # calculate direction
            # KKT vector - r(x, lambda)
            grad_loss_x = self._grad_loss_x_fn(x_k)
            c_vals = self._eval_constraints(x_k)
            kkt_state = jnp.concatenate((grad_loss_x, c_vals))

            # Find a preconditioning matrix B for a gradient direction
            if self._direction == Direction.EXACT_NEWTON:
                B_k = self._hess_lagr_xx_fn(x_k, m_k)
            elif self._direction == Direction.GAUSS_NEWTON:
                raise NotImplementedError
                # B_k = grad_loss_x @ jnp.transpose(grad_loss_x)
            elif self._direction == Direction.STEEPEST_DESCENT:
                B_k = jnp.eye(N=self._x_dims)
            else:
                raise NotImplementedError

            # TODO check H for PD + regularization (Powell's trick)
            B_k_is_pd = None
            if (self._direction != Direction.STEEPEST_DESCENT
                    and self._reg != HessianRegularization.NONE):
                # TODO verify that Cholesky check is faster
                B_k_is_pd = self._is_pd_matrix(B_k)
                if not B_k_is_pd:
                    if self._reg == HessianRegularization.EIGEN_DELTA:
                        delta = 1e-5
                        eig_vals, eig_vecs = jnp.linalg.eigh(B_k)
                        eig_vals_modified = eig_vals.at[eig_vals < delta].set(delta)
                        B_k = eig_vecs @ jnp.diag(eig_vals_modified) @ jnp.transpose(eig_vecs)
                    elif self._reg == HessianRegularization.EIGEN_FLIP:
                        delta = 1e-5
                        eig_vals, eig_vecs = jnp.linalg.eigh(B_k)
                        eig_vals_modified = jnp.array([flip_eig(e, delta) for e in eig_vals])
                        B_k = eig_vecs @ jnp.diag(eig_vals_modified) @ jnp.transpose(eig_vecs)
                    else:
                        # TODO modified Cholesky
                        raise NotImplementedError

            # Form KKT matrix, r'(x, lambda)
            kkt_matrix = kkt_matrix.at[:self._x_dims, :self._x_dims].set(B_k)
            constr_grad_x = self._eval_constraint_gradients(x_k)
            kkt_matrix = kkt_matrix.at[:self._x_dims, self._x_dims:].set(constr_grad_x)
            kkt_matrix = kkt_matrix.at[self._x_dims:, :self._x_dims].set(jnp.transpose(constr_grad_x))

            # Note: state is multiplied by (-1)
            d_k = jnp.linalg.solve(kkt_matrix, -kkt_state)

            # calculate step size (line search)
            # curr_x, grad_loss_x, direction
            alpha_k = self._step_size_fn(x_k=x_k, grad_loss_x=grad_loss_x, direction=d_k)

            # save cache + logs
            loss = self._loss_fn(x_k)
            x_dir_norm = jnp.max(jnp.abs(d_k[:self._x_dims]))
            cache_item = CacheItem(x_k, m_k, loss, alpha_k, x_dir_norm, B_k_is_pd, conv_penalty, self._sigma)
            self._cache[k] = cache_item
            self._logger.info(self._get_log_str(k, cache_item))

            # update state
            x_k += alpha_k * d_k[:self._x_dims]
            m_k = (1 - alpha_k) * m_k + alpha_k * d_k[self._x_dims:]

            # TODO check update sigma
            self._sigma = jnp.max(jnp.abs(m_k)) + 0.1

            # check convergence
            converged, conv_penalty = self._convergence_fn(x_k, m_k)

            # increment counter
            k += 1

        # log and print last results
        loss = self._loss_fn(x_k)
        cache_item = CacheItem(x_k, m_k, loss, .0, .0, None, conv_penalty, self._sigma)
        self._cache[k] = cache_item
        self._logger.info(self._get_log_str(k, cache_item))

        # fill additional info
        info = (converged, loss, k, self._cache)

        return x_k, info


def flip_eig(e, delta):
    if jnp.isclose(e, 0.):
        return delta
    elif e < 0:
        return -e
    else:
        return e
