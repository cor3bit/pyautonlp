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

            u_min: float,
            u_max: float,

            # loss_fn: Callable = None,
            # eq_constr: List[Callable] = None,
            # ineq_constr: List[Callable] = None,

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
        self._logger.info(f'Dimensions of the state vector x(t): {self._x_dims}.')

        self._u_dims = u_dims
        self._logger.info(f'Dimensions of the control vector u(t): {self._u_dims}.')

        self._x_pen = x_penalty
        self._u_pen = u_penalty

        self._u_min = u_min
        self._u_max = u_max

        self._w_dims = self._n_steps * self._u_dims
        self._logger.info(f'Dimensions of the decision vector w: {self._w_dims}.')

        # constraints
        # equality constraints on the last state x(N)
        self._eq_mult_dims = self._x_dims
        self._logger.info(f'Dimensions of the equality multiplier vector g(w): {self._eq_mult_dims}.')

        # inequality constraint (u_max and u_min) for each control variable
        self._ineq_mult_dims = 2 * self._w_dims
        self._logger.info(f'Dimensions of the inequality multiplier vector h(w): {self._ineq_mult_dims}.')

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
        w_k = jnp.zeros((self._w_dims,), dtype=jnp.float32)

        # residuals R(w)
        r_dims = self._n_steps * (self._x_dims + self._u_dims) + self._x_dims
        self._logger.info(f'Dimensions of the residual vector R(w): {r_dims}.')

        # lagrange multipliers
        m_eq_k = jnp.zeros((self._eq_mult_dims,), dtype=jnp.float32)
        m_ineq_k = jnp.zeros((self._ineq_mult_dims,), dtype=jnp.float32)

        # constraints preprocessing
        grad_c_ineq_k = jnp.vstack(
            [
                jnp.diag(jnp.ones((self._w_dims,), jnp.float32)),
                jnp.diag(-jnp.ones((self._w_dims,), jnp.float32)),
            ]
        )
        w_u_max = jnp.full(shape=(self._w_dims,), fill_value=self._u_max, dtype=jnp.float32)
        w_u_min = jnp.full(shape=(self._w_dims,), fill_value=self._u_min, dtype=jnp.float32)

        # precompiles dynamics jacobians
        dfds_fn = jit(jacfwd(self._dynamics, argnums=0))
        dfdu_fn = jit(jacfwd(self._dynamics, argnums=1))

        # create control discretization grid
        time_grid_shooting = jnp.linspace(self._t0, self._tf, num=self._n_steps + 1)

        # run main SQP loop
        converged = False
        k = 0
        while k < self._max_iter:
            # logging
            self._logger.info(f'---------------- Iteration {k} ----------------')
            self._log_param(k, 'w', w_k)
            self._log_param(k, 'm_eq', m_eq_k)
            self._log_param(k, 'm_ineq', m_ineq_k)

            # STEP 1. Given current estimate of u, simulate the evolution of x
            # xs = [self._x0]
            x_i = self._x0

            R = []
            Jac_R = jnp.zeros((r_dims, self._w_dims), jnp.float32)

            # construct R and Jacobian of R
            dxdw_prev = jnp.zeros((self._x_dims, self._w_dims), jnp.float32)

            # convert w -> u
            # Note. w is flat, u is (n_steps-1)x(n_u)
            u_k = jnp.reshape(w_k, (self._n_steps, self._u_dims))

            for i, (t_start, t_end, u_i) in enumerate(zip(time_grid_shooting[:-1], time_grid_shooting[1:], u_k)):
                time_grid_integration = jnp.linspace(t_start, t_end, num=self._n_steps_internal + 1)

                x_full, info = integrate(self._dynamics, x_i, u_i, time_grid_integration,
                                         method=IntegrateMethod.EEULER, with_grads=True,
                                         dfds_fn=dfds_fn, dfdu_fn=dfdu_fn)
                x_j = x_full[-1]
                G_x = info['G_x']
                G_u = info['G_u'].reshape((-1))

                # R components: calc loss based on the starting values
                loss_x_i = (x_i - self._xf) * jnp.sqrt(self._x_pen)
                loss_u_i = u_i * jnp.sqrt(self._u_pen)
                R.append(loss_x_i)
                R.append(loss_u_i)

                # Jac_R components: chain rule, one row at a time
                diag_component = G_u * jnp.sqrt(self._x_pen)
                dxdw_next = G_x @ dxdw_prev
                dxdw_next = dxdw_next.at[:, i].set(diag_component)

                start_ind = (i+1) * (self._x_dims + self._u_dims)
                end_ind = start_ind + self._x_dims
                Jac_R = Jac_R.at[start_ind:end_ind, :].set(dxdw_next)

                # update x_i
                x_i = x_j

                dxdw_prev = dxdw_next

            self._log_param(k, 'x_N', x_i)

            # last time step - x(N)
            loss_x_i = (x_i - self._xf) * jnp.sqrt(self._x_pen)
            R.append(loss_x_i)

            # convert to one vector
            R = jnp.concatenate(R)
            assert R.shape[0] == r_dims

            loss = 0.5 * R.T @ R
            self._log_param(k, 'loss', loss)

            grad_loss = Jac_R.T @ R
            B_k = Jac_R.T @ Jac_R

            c_eq_k = x_i - self._xf

            # undo *sqrt(Q) for pure dx(n)/dw
            a = Jac_R[-self._x_dims:, :]
            b = jnp.repeat(jnp.sqrt(self._x_pen), self._w_dims).reshape((self._x_dims, self._w_dims))
            grad_c_eq_k = a / b

            c_ineq_k = jnp.concatenate([
                w_k - w_u_max,
                -w_k + w_u_min,
            ])

            converged, kkt_viol = self._shooting_converged(k, grad_loss, m_eq_k, m_ineq_k, c_eq_k,
                                                           c_ineq_k, grad_c_eq_k, grad_c_ineq_k)
            self._log_param(k, 'penalty', kkt_viol)

            # TODO check that breaks the loop
            if converged:
                break

            # find direction by solving QP
            c_k = jnp.concatenate([c_eq_k, c_ineq_k])
            grad_c_k = jnp.vstack([grad_c_eq_k, grad_c_ineq_k])

            # TODO REMOVE DEBUG
            np_R = np.array(R)
            np_Jac_R = np.array(Jac_R)
            np_B_k = np.array(B_k)
            np_grad_loss = np.array(grad_loss)
            np_c_k = np.array(c_k)
            np_grad_c_k = np.array(grad_c_k)


            d_k = self._solve_qp(B_k, grad_loss, c_k, grad_c_k.T)
            self._log_param(k, 'd', d_k)

            # calculate step size (line search)
            # TODO recheck backtracking
            alpha_k = self._backtrack(x_k=w_k, grad_loss_x=grad_loss, direction=d_k, max_iter=7)
            self._log_param(k, 'sigma', self._sigma)
            self._log_param(k, 'alpha', alpha_k)

            # update controls and multipliers
            # TODO recheck for shooting
            w_k += alpha_k * d_k[:self._x_dims]
            m_k = (1 - alpha_k) * m_k + alpha_k * d_k[self._x_dims:]
            self._sigma = jnp.max(jnp.abs(m_k)) + 0.1

            # increment counter
            k += 1

        if k < self._max_iter:
            self._logger.info(f'Converged after {k} iterations!')
        else:
            self._logger.info(f'Maximum number of {k} iterations reached.')

        # fill additional info
        info = (converged, k, self._cache)

        return w_k, info

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

    def _shooting_converged(self, k, grad_loss, m_eq_k, m_ineq_k, c_eq_k, c_ineq_k, grad_c_eq_k, grad_c_ineq_k):
        max_c_eq = jnp.max(jnp.abs(c_eq_k))
        max_c_ineq = jnp.max(jnp.clip(c_ineq_k, a_min=0))
        grad_lagr = grad_loss + m_eq_k @ grad_c_eq_k + m_ineq_k @ grad_c_ineq_k
        max_lagr_grad = jnp.max(jnp.abs(grad_lagr))

        self._log_param(k, 'c_eq violation', max_c_eq, save=False)
        self._log_param(k, 'c_ineq violation', max_c_ineq, save=False)
        self._log_param(k, 'grad_Lagrangian', max_lagr_grad, save=False)

        max_viol = jnp.maximum(jnp.maximum(max_c_eq, max_c_ineq), max_lagr_grad)

        return max_viol <= self._tol, max_viol
