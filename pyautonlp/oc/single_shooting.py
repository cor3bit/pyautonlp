import logging
from typing import List, Tuple, Callable, Optional, Union

import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, jit
from quadprog import solve_qp

from pyautonlp.constants import *
from pyautonlp.constr.constr_solver import ConstrainedSolver
from pyautonlp.oc.integrators import integrate


class SingleShooting(ConstrainedSolver):
    def __init__(
            self,

            dynamics: Callable,
            t0: float,
            tf: float,
            x0: jnp.ndarray,
            xf: jnp.ndarray,
            u_dims: int,

            x_penalty: jnp.ndarray,
            u_penalty: jnp.ndarray,

            # loss_fn: Callable = None,
            eq_constr: List[Callable] = None,
            ineq_constr: List[Callable] = None,

            n_steps: int = 20,
            n_steps_internal: int = 10,

            beta: float = 0.5,  # relevant for Backtracking line search
            gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            sigma: float = 1.0,  # relevant for Backtracking + Merit Function line search

            conv_tol: float = 1e-8,
            max_iter: int = 50,

            verbose: bool = False,
            **kwargs
    ):
        # logger
        self._logger = logging.getLogger('single_shooting')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # loss
        # self._loss_fn = loss_fn

        # oc params
        self._dynamics = dynamics
        self._t0 = t0
        self._tf = tf
        self._x0 = x0
        self._xf = xf
        self._n_steps = n_steps
        self._n_steps_internal = n_steps_internal

        self._x_dims = x0.shape[0]
        self._logger.info(f'Dimensions of the state vector: {self._x_dims}.')

        self._u_dims = u_dims
        self._logger.info(f'Dimensions of the control vector: {self._u_dims}.')

        self._x_pen = x_penalty
        self._u_pen = u_penalty

        # constraints
        self._eq_constr = eq_constr
        self._ineq_constr = ineq_constr
        self._constr_fns = []
        if eq_constr is not None and eq_constr:
            self._constr_fns.extend(eq_constr)
        if ineq_constr is not None and ineq_constr:
            self._constr_fns.extend(ineq_constr)
        assert self._constr_fns

        self._eq_mult_dims = 0 if eq_constr is None else len(eq_constr)
        self._logger.info(f'Dimensions of the equality multiplier vector: {self._eq_mult_dims}.')

        self._ineq_mult_dims = 0 if ineq_constr is None else len(ineq_constr)
        self._logger.info(f'Dimensions of the inequality multiplier vector: {self._ineq_mult_dims}.')
        self._multiplier_dims = self._eq_mult_dims + self._ineq_mult_dims

        self._initial_multipliers = jnp.zeros(shape=(self._multiplier_dims,), dtype=jnp.float32)
        self._logger.info(f'Dimensions of the multiplier vector: {self._multiplier_dims}.')

        # convergence
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma

        # save intermediate step info
        self._cache = {}
        self._step_cache = {}

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        if self._u_dims != 1:
            raise NotImplementedError

        # control variable, flatten array of u_0, u_1, ..., u_{N-1}
        w_dims = self._n_steps * self._u_dims
        w_k = jnp.zeros((w_dims,), dtype=jnp.float32)
        self._logger.info(f'Initial controls are: {w_k}.')

        r_dims = self._n_steps * (self._x_dims + self._u_dims) + self._x_dims

        # precompiles dynamics jacobians
        dfds_fn = jit(jacfwd(self._dynamics, argnums=0))
        dfdu_fn = jit(jacfwd(self._dynamics, argnums=1))

        # create control discretization grid
        time_grid_shooting = jnp.linspace(self._t0, self._tf, num=self._n_steps + 1)

        # run main SQP loop
        converged = False
        k = 0
        while (not converged) and (k < self._max_iter):

            # STEP 1. Given current estimate of u, simulate the evolution of x
            # xs = [self._x0]
            x_i = self._x0

            R = []
            Jac_R = []

            # TODO construct R and Jacobian of R
            dxdw = jnp.zeros((self._x_dims, w_dims), jnp.float32)

            n_steps_internal = 10
            for t_start, t_end, u_i in zip(time_grid_shooting[:-1], time_grid_shooting[1:], w_k):
                time_grid_integration = jnp.linspace(t_start, t_end, num=n_steps_internal + 1)

                x_full, info = integrate(self._dynamics, x_i, u_i, time_grid_integration,
                                         method=IntegrateMethod.EEULER, with_grads=True,
                                         dfds_fn=dfds_fn, dfdu_fn=dfdu_fn)
                x_j = x_full[-1]
                G_x = info['G_x']
                G_u = info['G_u']

                # calc loss based on the starting values
                loss_x_i = (x_i - self._xf) * jnp.sqrt(self._x_pen)
                loss_u_i = u_i * jnp.sqrt(self._u_pen)
                R.append(loss_x_i)
                R.append(loss_u_i)

                # update x_i
                x_i = x_j

            # last time step - x(N)
            loss_x_i = (x_i - self._xf) * jnp.sqrt(self._x_pen)
            R.append(loss_x_i)

            # convert to one vector
            R = jnp.vstack(R)
            assert R.shape[0] == r_dims

            loss = 0.5 * R.T @ R
            grad_loss = Jac_R.T @ R
            B_k = Jac_R.T @ Jac_R

            # find direction by solving QP
            c_k = self._eval_constraints(x_k)
            constr_grad_x = self._eval_constraint_gradients(x_k)

            d_k = self._solve_qp(B_k, grad_loss, c_k, constr_grad_x)

            # calculate step size (line search)
            alpha_k = self._backtrack(x_k=w_k, grad_loss_x=grad_loss, direction=d_k, max_iter=7)

            # save cache + logs
            # loss = self._loss_fn(x_k)
            self._step_cache.update({
                'k': k,
                'alpha': alpha_k, 'sigma': self._sigma,
                'loss': loss, 'penalty': conv_penalty, 'w': w_k,
            })
            self._cache[k] = dict(self._step_cache)
            self._log_step()

            # update state
            # TODO dims
            w_k += alpha_k * d_k[:self._x_dims]
            m_k = (1 - alpha_k) * m_k + alpha_k * d_k[self._x_dims:]

            self._logger.info(f'Updated x_k: {x_k}.')

            # TODO check update sigma
            self._sigma = jnp.max(jnp.abs(m_k)) + 0.1

            # check convergence
            # TODO move to beginning
            converged, conv_penalty = self._shooting_converged(x_k, m_k)

            # increment counter
            k += 1

        # log and print last results
        # loss = self._loss_fn(x_k)
        self._step_cache.update({
            'k': k, 'sigma': self._sigma,
            'loss': loss, 'penalty': conv_penalty, 'x': x_k,
        })
        self._cache[k] = dict(self._step_cache)
        self._log_step()

        # fill additional info
        info = (converged, loss, k, self._cache)

        return x_k, info

    def _solve_qp(self, B_k, grad_loss_x, c_k, constr_grad_x):
        # convert to numpy
        G = np.array(B_k, dtype=np.double)
        a = np.array(grad_loss_x, dtype=np.double)
        b = np.array(c_k, dtype=np.double)
        C = np.array(constr_grad_x, dtype=np.double)

        # solve QP
        xf, f, xu, iters, lagr, iact = solve_qp(G, a, C, b, meq=self._eq_mult_dims)

        # translate back to JAX
        d_k = jnp.array(np.concatenate((xf, lagr)))
        d_k *= -1.

        return d_k

    def _shooting_converged(self, x_k, m_k, **kwargs):
        # TODO optimize
        max_c_eq = jnp.max(jnp.abs(self._eval_eq_constraints(x_k)))

        max_c_ineq = jnp.max(jnp.clip(self._eval_ineq_constraints(x_k), a_min=0))

        max_lagr_grad = jnp.max(jnp.abs(self._grad_lagr_x_fn(x_k, m_k)))

        max_viol = jnp.maximum(jnp.maximum(max_c_eq, max_c_ineq), max_lagr_grad)

        return max_viol <= self._tol, max_viol