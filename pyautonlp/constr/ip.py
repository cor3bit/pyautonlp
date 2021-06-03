import logging
from typing import List, Tuple, Callable, Optional, Union

import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacrev, jit

from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization, KKTForm
from pyautonlp.constr.constr_solver import ConstrainedSolver, CacheItem


class IP(ConstrainedSolver):
    def __init__(
            self,
            loss_fn: Callable,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,
            guess: Union[Tuple, jnp.ndarray] = None,
            kkt_form: str = KKTForm.FULL,
            direction: str = Direction.STEEPEST_DESCENT,
            reg: str = HessianRegularization.NONE,
            line_search: str = LineSearch.CONST,
            alpha: float = 0.01,  # relevant for Constant step line search
            beta: float = 0.5,  # relevant for Backtracking line search
            gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            sigma: float = 1.0,  # relevant for Backtracking + Merit Function line search
            conv_criteria: str = ConvergenceCriteria.KKT_VIOLATION,
            conv_tol: float = 1e-8,
            max_iter: int = 50,
            verbose: bool = False,
            **kwargs
    ):
        # logger
        self._logger = logging.getLogger('sqp_solver')
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
        self._eq_constr = [] if eq_constr is None else eq_constr
        self._ineq_constr = [] if eq_constr is None else ineq_constr
        self._constr_fns = self._eq_constr + self._ineq_constr

        assert self._ineq_constr
        assert self._constr_fns

        self._eq_mult_dims = len(self._eq_constr)
        self._logger.info(f'Dimensions of the equality multiplier vector: {self._eq_mult_dims}.')

        self._ineq_mult_dims = len(self._ineq_constr)
        self._logger.info(f'Dimensions of the equality multiplier vector: {self._ineq_mult_dims}.')
        self._multiplier_dims = self._eq_mult_dims + self._ineq_mult_dims

        self._initial_eq_mult = jnp.zeros(shape=(self._eq_mult_dims,), dtype=jnp.float32)
        self._initial_ineq_mult = jnp.ones(shape=(self._ineq_mult_dims,), dtype=jnp.float32)
        self._initial_slack = jnp.ones(shape=(self._ineq_mult_dims,), dtype=jnp.float32)

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

        # grad for loss & constraints
        # compile with JAX in advance
        self._grad_loss_x_fn = grad(self._loss_fn)  # N-by-1
        self._grad_eq_constr_x_fns = [grad(f) for f in self._eq_constr]
        self._grad_ineq_constr_x_fns = [grad(f) for f in self._ineq_constr]
        self._grad_constr_x_fns = self._grad_eq_constr_x_fns + self._grad_ineq_constr_x_fns

        # grad for lagrangians
        self._direction = direction
        if direction == Direction.EXACT_NEWTON:
            self._grad_lagr_x_fn = grad(self._ip_lagrangian)
            self._hess_lagr_xx_fn = self._hessian_exact(self._ip_lagrangian)  # N-by-N
        elif direction == Direction.BFGS:
            self._grad_lagr_x_fn = grad(self._ip_lagrangian)
            self._hess_lagr_xx_fn = self._hessian_bfgs_approx  # N-by-N
        elif direction == Direction.GAUSS_NEWTON:
            self._grad_lagr_x_fn = grad(self._ip_lagrangian)
            self._hess_lagr_xx_fn = self._hessian_gn_approx  # N-by-N
        elif direction == Direction.STEEPEST_DESCENT:
            self._hess_lagr_xx_fn = self._hessian_sd_approx  # N-by-N
        else:
            raise NotImplementedError

        self._reg = reg

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        tau = 1.

        x_k = self._initial_x
        self._logger.info(f'Initial guess is: {x_k}.')

        eq_m_k = self._initial_eq_mult
        ineq_m_k = self._initial_ineq_mult
        s_k = self._initial_slack
        # self._logger.info(f'Initial multipliers are: {m_k}.')

        # dimensions
        n_x = self._x_dims
        n_g = self._eq_mult_dims
        n_h = self._ineq_mult_dims
        kkt_n = n_x + n_g + 2 * n_h

        kkt_matrix = jnp.zeros(shape=(kkt_n, kkt_n), dtype=jnp.float32)
        identity = jnp.eye(n_h, dtype=jnp.float32)

        converged, conv_penalty = self._ip_convergence_fn(x_k, eq_m_k, ineq_m_k, s_k, tau)
        k = 0
        B_prev = None
        x_prev = None
        g_prev = None
        g_k = None
        while (not converged) and (k < self._max_iter):
            # calculate B_k
            if self._direction == Direction.EXACT_NEWTON:
                B_k = self._hess_lagr_xx_fn(x_k, eq_m_k, ineq_m_k, s_k, tau)
            elif self._direction == Direction.BFGS:
                g_k = self._grad_lagr_x_fn(x_k, eq_m_k, ineq_m_k)
                B_k = self._hess_lagr_xx_fn(B_prev, g_k, g_prev, x_k, x_prev)
            elif self._direction == Direction.GAUSS_NEWTON:
                g_k = self._grad_lagr_x_fn(x_k, eq_m_k, ineq_m_k)
                B_k = self._hess_lagr_xx_fn(g_k)
            else:
                B_k = self._hess_lagr_xx_fn()

            # ensure B_k is pd
            B_k_is_pd = None
            if (self._direction != Direction.STEEPEST_DESCENT
                    and self._direction != Direction.BFGS
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
                        eig_vals_modified = jnp.array([self._flip_eig(e, delta) for e in eig_vals])
                        B_k = eig_vecs @ jnp.diag(eig_vals_modified) @ jnp.transpose(eig_vecs)
                    else:
                        # TODO modified Cholesky
                        raise NotImplementedError

            # build KKT matrix, r'(x, lambda, mu, s)
            # insert B_k
            kkt_matrix = kkt_matrix.at[:n_x, :n_x].set(B_k)

            # insert [nabla_g, nabla_h]
            constr_grad_x = self._eval_constraint_gradients(x_k)
            kkt_matrix = kkt_matrix.at[:n_x, n_x:n_x + n_g + n_h].set(constr_grad_x)
            kkt_matrix = kkt_matrix.at[n_x:n_x + n_g + n_h, :n_x].set(jnp.transpose(constr_grad_x))

            # insert I
            kkt_matrix = kkt_matrix.at[n_x + n_g:n_x + n_g + n_h, n_x + n_g + n_h:].set(identity)

            # insert S_k
            S_k = jnp.diag(s_k)
            kkt_matrix = kkt_matrix.at[n_x + n_g + n_h:, n_x + n_g:n_x + n_g + n_h].set(S_k)

            # insert M_k
            M_k = jnp.diag(ineq_m_k)
            kkt_matrix = kkt_matrix.at[n_x + n_g + n_h:, n_x + n_g + n_h:].set(M_k)

            # build KKT vector, r(x, lambda)
            grad_lagr_x = self._grad_lagr_x_fn(x_k, eq_m_k, ineq_m_k, s_k, tau)
            c_k = self._eval_constraints_with_slack(x_k, s_k)
            slack_component = M_k @ s_k - tau
            kkt_state = jnp.concatenate((grad_lagr_x, c_k, slack_component))

            # calculate direction
            # Note: state is multiplied by (-1)
            d_k = jnp.linalg.solve(kkt_matrix, -kkt_state)

            # calculate step size (line search)
            alpha_k = None # self._step_size_fn(x_k=x_k, grad_loss_x=grad_loss_x, direction=d_k)

            # update params for BFGS
            if self._direction == Direction.BFGS:
                B_prev = B_k
                x_prev = x_k
                g_prev = g_k

            # save cache + logs
            loss = self._loss_fn(x_k)
            x_dir_norm = jnp.max(jnp.abs(d_k[:self._x_dims]))
            cache_item = CacheItem(x_k, m_k, loss, alpha_k, x_dir_norm, B_k_is_pd, conv_penalty, self._sigma)
            self._cache[k] = cache_item
            self._logger.info(self._get_log_str(k, cache_item))

            # update variables
            x_k += alpha_k * d_k[:self._x_dims]
            m_k = (1 - alpha_k) * m_k + alpha_k * d_k[self._x_dims:]

            # TODO check update sigma
            self._sigma = jnp.max(jnp.abs(m_k)) + 0.1

            # TODO update tau

            # check convergence
            converged, conv_penalty = self._ip_convergence_fn(x_k, eq_m_k, ineq_m_k, s_k, tau)

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

    def _ip_lagrangian(self, x, eq_mult, ineq_mult, slack, tau):
        loss_x = self._loss_fn(x)
        barrier = -tau * jnp.sum(jnp.log(slack))
        eq_penalty = jnp.sum(jnp.array([c_fn(x) for c_fn in self._eq_constr]) * eq_mult)
        ineq_penalty = jnp.sum((jnp.array([c_fn(x) for c_fn in self._ineq_constr]) + slack) * ineq_mult)
        return loss_x + barrier + eq_penalty + ineq_penalty

    def _ip_convergence_fn(self, x_k, eq_m_k, ineq_m_k, s_k, tau):
        max_c_eq = jnp.max(jnp.abs(self._eval_eq_constraints(x_k)))
        max_c_ineq = jnp.max(jnp.clip(self._eval_ineq_constraints(x_k), a_min=0))
        max_lagr_grad = jnp.max(jnp.abs(self._grad_lagr_x_fn(x_k, eq_m_k, ineq_m_k, s_k, tau)))

        max_viol = jnp.max(jnp.array([max_c_eq, max_c_ineq, max_lagr_grad, tau]))

        return max_viol <= self._tol, max_viol
