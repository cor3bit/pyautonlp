import logging
from time import perf_counter
from typing import List, Tuple, Callable, Optional, Union
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jit
import qpsolvers

from pyautonlp.constants import *
from pyautonlp.viz import Visualizer
from pyautonlp.constr.constr_solver import ConstrainedSolver
from pyautonlp.oc.integrators import integrate, integrate_with_ad


class MultipleShooting(ConstrainedSolver):
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

            n_steps: int = 20,
            n_steps_internal: int = 10,

            beta: float = 0.5,  # relevant for Backtracking line search
            gamma: float = 0.1,  # relevant for Backtracking + Armijo line search
            sigma: float = 1.0,  # relevant for Backtracking + Merit Function line search

            conv_tol: float = 1e-4,
            max_iter: int = 50,

            verbose: bool = False,
            visualize: bool = False,
            visualize_n: int = 10,
            save_plot_dir: Optional[str] = None,
            **kwargs
    ):
        # logger
        self._logger = logging.getLogger('multiple_shooting')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # oc params
        self._dynamics = dynamics
        self._t0 = t0
        self._tf = tf
        self._x0 = x0
        self._xf = xf
        self._n_steps = n_steps
        self._n_steps_internal = n_steps_internal

        self._n_x = x0.shape[0]
        self._logger.info(f'Dimensions of the state vector x(t): {self._n_x}.')

        self._n_u = u_dims
        self._logger.info(f'Dimensions of the control vector u(t): {self._n_u}.')

        self._x_pen = x_penalty
        self._u_pen = u_penalty

        self._u_min = u_min
        self._u_max = u_max

        self._n_w = self._n_steps * (self._n_u + self._n_x) + self._n_x
        self._logger.info(f'Dimensions of the decision vector w: {self._n_w}.')

        # pre-built arrays
        self._w_penalty, self._w_ref = self._build_w_penalty_ref()
        self._w_u_max = jnp.full(shape=(self._n_u * self._n_steps,), fill_value=self._u_max, dtype=jnp.float32)
        self._w_u_min = jnp.full(shape=(self._n_u * self._n_steps,), fill_value=self._u_min, dtype=jnp.float32)

        # constraints
        # equality constraints on the last state x(N)
        self._n_eq_mult = self._n_x * (self._n_steps + 2)
        self._logger.info(f'Dimensions of the equality multiplier vector g(w): {self._n_eq_mult}.')

        # inequality constraint (u_max and u_min) for each control variable
        self._n_ineq_mult = 2 * self._n_steps * self._n_u
        self._logger.info(f'Dimensions of the inequality multiplier vector h(w): {self._n_ineq_mult}.')

        # convergence
        self._max_iter = max_iter
        self._tol = conv_tol

        # learning rate
        self._beta = beta
        self._gamma = gamma
        self._sigma = sigma

        # save intermediate step info
        self._cache = {}

        # viz
        self._viz = visualize
        self._visualize_n = visualize_n
        self._save_plot_dir = save_plot_dir

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        if self._n_u != 1:
            raise NotImplementedError

        # timing
        solve_t0 = perf_counter()

        # control variable, flatten array of x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N
        w_k = jnp.zeros((self._n_w,), dtype=jnp.float32)

        # lagrange multipliers
        m_eq_k = jnp.zeros((self._n_eq_mult,), dtype=jnp.float32)
        m_ineq_k = jnp.zeros((self._n_ineq_mult,), dtype=jnp.float32)

        # constraints preprocessing
        grad_c_ineq_k = self._build_grad_ineq_matrix()

        # Hessian preprocessing
        B_k = jnp.diag(self._w_penalty)

        # precompiles dynamics jacobians
        dfds_fn = jit(jacfwd(self._dynamics, argnums=0))
        dfdu_fn = jit(jacfwd(self._dynamics, argnums=1))

        # create control discretization grid
        tg_shooting = jnp.linspace(self._t0, self._tf, num=self._n_steps + 1)

        # pure loss function for backtracking
        self._loss_fn = partial(self._ms_loss_fn, k=0, with_grads=False)

        # run main SQP loop
        converged = False
        k = 0

        self._log_param(k, 'n_steps', self._n_steps, display=False)
        self._log_param(k, 'x0', self._x0, display=False)
        self._log_param(k, 'xf', self._xf, display=False)

        while k < self._max_iter:
            self._logger.info(f'---------------- Iteration {k} ----------------')

            # given w_k, calculate evolution of x
            loss, grad_loss = self._ms_loss_fn(w_k, k, with_grads=True)

            # construct constraints matrices
            c_eq_k, grad_c_eq_k = self._eval_eq_constraints_with_grads(
                w_k, k, tg_shooting, dfds_fn, dfdu_fn)

            c_ineq_k = self._eval_ineq_constraints(x=w_k, time_grid_shooting=tg_shooting)

            # convergence
            converged = self._ms_converged(k, grad_loss, m_eq_k, c_eq_k, grad_c_eq_k)

            if converged:
                self._logger.info(f'Converged after {k} iterations!')
                break

            # find direction by solving QP
            c_k = jnp.concatenate([c_eq_k, c_ineq_k])
            grad_c_k = jnp.vstack([grad_c_eq_k, grad_c_ineq_k])

            # TODO REMOVE, DEBUG-ONLY
            # np_B_k = np.array(B_k)
            # np_grad_loss = np.array(grad_loss)
            # np_c_eq_k = np.array(c_eq_k)
            # np_grad_c_eq_k = np.array(grad_c_eq_k)
            # np_c_ineq_k = np.array(c_ineq_k)
            # np_grad_c_ineq_k = np.array(grad_c_ineq_k)
            # np_c_k = np.array(c_k)
            # np_grad_c_k = np.array(grad_c_k)

            # TODO relax if infeasible
            try:
                d_k = self._solve_qp(B_k, grad_loss, c_k, grad_c_k.T, self._n_eq_mult)
                # self._log_param(k, 'd', d_k[:self._w_dims])
            except Exception as e:
                self._logger.warning(f'QP is infeasible! Failed with {e}.')
                self._logger.info('Trying unbounded solver.')
                # TODO return m_k
                # d_k = self._solve_infeasible_qp(B_k, grad_loss, c_eq_k, c_ineq_k, grad_c_eq_k, grad_c_ineq_k)
                # self._log_param(k, 'd2', d_k)

                # raise NotImplementedError
                d_k = self._solve_qp(B_k, grad_loss, c_eq_k, grad_c_eq_k.T, self._n_eq_mult)

            self._log_param(k, 'max_d', jnp.max(d_k), save=False)
            self._log_param(k, 'min_d', jnp.min(d_k), save=False)

            # calculate step size (line search)
            alpha_k = self._ms_backtrack(w_k=w_k, d_k=d_k, loss=loss, grad_loss=grad_loss, c_eq_k=c_eq_k,
                                         time_grid_shooting=tg_shooting, max_iter=10)

            self._log_param(k, 'sigma', self._sigma)
            self._log_param(k, 'alpha', alpha_k)

            # update controls and multipliers
            w_k += alpha_k * d_k[:self._n_w]

            m_eq_ind_end = self._n_w + self._n_eq_mult
            m_eq_k = (1 - alpha_k) * m_eq_k + alpha_k * d_k[self._n_w:m_eq_ind_end]
            # m_ineq_k = (1 - alpha_k) * m_ineq_k + alpha_k * d_k[m_eq_ind_end:]

            self._sigma = jnp.max(jnp.abs(jnp.concatenate([m_eq_k, m_ineq_k]))) + 0.1

            # increment counter
            k += 1

        if k == self._max_iter:
            self._logger.info(f'Maximum number of {k} iterations reached.')

        # fill additional info
        info = (converged, k, self._cache)

        # charts
        if self._viz:
            Visualizer(
                solver_type=SolverType.MULT_SHOOTING,
                solver_caches=[self._cache],
                save_dir=self._save_plot_dir,
                x1_bounds=(self._u_min, self._u_max),
                n=self._visualize_n,
            ).plot_shooting()

        self._logger.info(f'Total solving time: {(perf_counter() - solve_t0) / 60.:.3f} min.')

        return w_k, info

    def _build_w_penalty_ref(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        p = []
        r = []
        u_ref = jnp.zeros((self._n_u,), dtype=jnp.float32)
        for _ in range(self._n_steps):
            # penalties
            p.append(self._x_pen)
            p.append(self._u_pen)

            # ref
            r.append(self._xf)
            r.append(u_ref)

        p.append(self._x_pen)
        r.append(self._xf)

        return jnp.concatenate(p), jnp.concatenate(r)

    def _build_grad_ineq_matrix(self) -> jnp.ndarray:
        grad_c_ineq_k = jnp.zeros((self._n_ineq_mult, self._n_w), jnp.float32)

        step = self._n_x + self._n_u
        for i, i_x in enumerate(range(self._n_x, self._n_w, step)):
            i_y = i * 2 * self._n_u
            grad_c_ineq_k = grad_c_ineq_k.at[i_y, i_x].set(1.)
            grad_c_ineq_k = grad_c_ineq_k.at[i_y + 1, i_x].set(-1.)

        return grad_c_ineq_k

    def _split_by_x_u(self, w_k: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n = self._n_steps
        n_x = self._n_x
        n_u = self._n_u

        # extract x
        x_vals = list(range(n_x))
        x_ind = [x_val + i * n_x + i * n_u for i in range(n + 1) for x_val in x_vals]
        w_k_x = jnp.take(w_k, x_ind)
        xs = jnp.reshape(w_k_x, (self._n_steps + 1, self._n_x))

        # extract u
        u_vals = list(range(n_x, n_x + n_u))
        u_ind = [u_val + i * n_x + i * n_u for i in range(n) for u_val in u_vals]
        w_k_u = jnp.take(w_k, u_ind)
        us = jnp.reshape(w_k_u, (self._n_steps, self._n_u))

        return xs, us

    def _ms_loss_fn(
            self,
            w_k: jnp.ndarray,
            k: int,
            with_grads: bool = False,
    ) -> Union[float, Tuple[float, jnp.ndarray]]:
        res = w_k - self._w_ref
        grad_loss = self._w_penalty * res
        loss = 0.5 * jnp.sum(grad_loss * res)

        if with_grads:
            self._log_param(k, 'loss', loss, save=True)
            return loss, grad_loss

        return loss

    def _eval_eq_constraints_with_grads(
            self,
            w_k: jnp.ndarray,
            k: int,
            time_grid_shooting: jnp.ndarray,
            dfds_fn: Callable,
            dfdu_fn: Callable,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self._logger.info('Equality constraints evaluation started.')
        t1 = perf_counter()

        # convert w -> x, u
        # Note. w is flat, x is (n_steps+1)x(n_x), u is (n_steps)x(n_u)
        xs, us = self._split_by_x_u(w_k)
        self._log_param(k, 'u', us, display=False)

        # initialize G(w) and grad_G(w) to be filled later
        c_eq_k_parts = [xs[0] - self._x0]

        diag_component = jnp.eye(self._n_x, dtype=jnp.float32)
        grad_c_eq_k = jnp.zeros((self._n_eq_mult, self._n_w), jnp.float32)
        grad_c_eq_k = grad_c_eq_k.at[:self._n_x, :self._n_x].set(diag_component)

        f_x = None
        for i, (t_start, t_end, x_i, x_j, u_i) in enumerate(zip(
                time_grid_shooting[:-1], time_grid_shooting[1:], xs[:-1], xs[1:], us)):
            tg_int = jnp.linspace(t_start, t_end, num=self._n_steps_internal + 1)

            x_full, info = integrate(self._dynamics, x_i, u_i, tg_int,
                                     method=IntegrateMethod.RK4, with_grads=True,
                                     dfds_fn=dfds_fn, dfdu_fn=dfdu_fn)
            f_x = x_full[-1]
            G_x = info['G_x']
            G_u = info['G_u'].reshape((-1))

            # fill G(w)
            c_eq_k_parts.append(f_x - x_j)

            # fill G_x, G_u
            ind_x_axis = i * (self._n_x + self._n_u)
            ind_y_axis = (i + 1) * self._n_x

            grad_c_eq_k = grad_c_eq_k.at[ind_y_axis:ind_y_axis + self._n_x, ind_x_axis:ind_x_axis + self._n_x].set(G_x)
            grad_c_eq_k = grad_c_eq_k.at[ind_y_axis:ind_y_axis + self._n_x, ind_x_axis + self._n_x].set(G_u)

            # fill diag(-1)
            diag_ind = ind_x_axis + self._n_x + self._n_u
            grad_c_eq_k = grad_c_eq_k.at[ind_y_axis:ind_y_axis + self._n_x, diag_ind:diag_ind + self._n_x].set(
                -diag_component)

        c_eq_k_parts.append(xs[-1] - self._xf)
        grad_c_eq_k = grad_c_eq_k.at[-self._n_x:, -self._n_x:].set(diag_component)

        self._log_param(k, 'f_x_N', f_x, save=False)
        self._logger.info(f'Equality constraints evaluation finished in {perf_counter() - t1:.3f} sec.')

        return jnp.concatenate(c_eq_k_parts), grad_c_eq_k

    def _eval_eq_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        # convert w -> x, u
        xs, us = self._split_by_x_u(x)

        # initialize G(w)
        c_eq_k_parts = [xs[0] - self._x0]

        tg = kwargs['time_grid_shooting']
        for i, (t_start, t_end, x_i, x_j, u_i) in enumerate(zip(tg[:-1], tg[1:], xs[:-1], xs[1:], us)):
            time_grid_integration = jnp.linspace(t_start, t_end, num=self._n_steps_internal + 1)
            x_full, info = integrate(self._dynamics, x_i, u_i, time_grid_integration, method=IntegrateMethod.RK4)
            f_x = x_full[-1]

            # fill G(w)
            c_eq_k_parts.append(f_x - x_j)

        c_eq_k_parts.append(xs[-1] - self._xf)

        return jnp.concatenate(c_eq_k_parts)

    def _eval_ineq_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        # convert w -> x, u
        xs, us = self._split_by_x_u(x)
        us_flat = us.reshape((-1))

        c_ineq_k = jnp.empty((2 * self._n_u * self._n_steps,), jnp.float32)
        c_ineq_k = c_ineq_k.at[::2].set(us_flat - self._w_u_max)
        c_ineq_k = c_ineq_k.at[1::2].set(-us_flat + self._w_u_min)

        return c_ineq_k

    def _eval_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        c_eq_k = self._eval_eq_constraints(x, **kwargs)
        c_ineq_k = self._eval_ineq_constraints(x, **kwargs)
        return jnp.concatenate([c_eq_k, c_ineq_k])

    def _ms_backtrack(
            self,
            w_k: jnp.ndarray,
            d_k: jnp.ndarray,
            loss: float,
            grad_loss: jnp.ndarray,
            c_eq_k: jnp.ndarray,
            time_grid_shooting: jnp.ndarray,
            initial_alpha: float = 1.,
            max_iter: int = 30,
            **kwargs
    ) -> float:
        self._logger.info('Backtracking started.')

        alpha = initial_alpha

        direction_w = d_k[:self._n_w]

        curr_loss = loss
        curr_merit_adj = self._ms_merit_adj(w_k, time_grid_shooting, c_eq_k)

        next_w_k = w_k + alpha * direction_w
        next_loss = self._loss_fn(next_w_k)
        next_merit_adj = self._ms_merit_adj(next_w_k, time_grid_shooting)

        armijo_adj = self._ms_armijo_adj(w_k, alpha, direction_w, grad_loss, time_grid_shooting, c_eq_k)

        n_iter = 0
        while (next_loss + next_merit_adj >= curr_loss + curr_merit_adj + armijo_adj) and (n_iter < max_iter):
            # update alpha
            alpha *= self._beta

            # update all alpha dependencies
            armijo_adj *= self._beta

            next_w_k = w_k + alpha * direction_w
            next_loss = self._loss_fn(next_w_k)
            next_merit_adj = self._ms_merit_adj(next_w_k, time_grid_shooting)

            n_iter += 1

        if n_iter == max_iter:
            self._logger.warning(f'Backtracking failed to find alpha after {max_iter} iterations!')
        else:
            self._logger.info(f'Backtracking converged after {n_iter} iterations.')

        return alpha

    def _ms_merit_adj(
            self,
            w_k: jnp.ndarray,
            time_grid_shooting: jnp.ndarray,
            c_eq_k: jnp.ndarray = None,
    ) -> float:
        if c_eq_k is None:
            c_eq_k = self._eval_eq_constraints(x=w_k, time_grid_shooting=time_grid_shooting)

        eq_norm = jnp.linalg.norm(c_eq_k, ord=1)

        ineq_norm = 0.
        # TODO skips inequality adjustment
        # c_ineq_k = self._eval_ineq_constraints(x=w_k)
        # ineq_norm = jnp.linalg.norm(jnp.clip(c_ineq_k, a_min=0), ord=1)

        return self._sigma * (eq_norm + ineq_norm)

    def _ms_armijo_adj(
            self,
            w_k: jnp.ndarray,
            alpha: float,
            direction_w: jnp.ndarray,
            grad_loss: jnp.ndarray,
            time_grid_shooting: jnp.ndarray,
            c_eq_k: jnp.ndarray,
    ) -> float:
        direct_deriv = jnp.dot(grad_loss, direction_w)
        return self._gamma * alpha * (direct_deriv - self._ms_merit_adj(w_k, time_grid_shooting, c_eq_k))

    def _ms_converged(
            self,
            k: int,
            grad_loss: jnp.ndarray,
            m_eq_k: jnp.ndarray,
            c_eq_k: jnp.ndarray,
            grad_c_eq_k: jnp.ndarray,
    ) -> bool:
        max_c_eq = jnp.max(jnp.abs(c_eq_k))

        grad_lagr = grad_loss + m_eq_k @ grad_c_eq_k
        max_lagr_grad = jnp.max(jnp.abs(grad_lagr))

        max_viol = jnp.maximum(max_c_eq, max_lagr_grad)

        self._log_param(k, 'max_c_eq', max_c_eq, save=True)
        self._log_param(k, 'max_grad_loss', jnp.max(jnp.abs(grad_loss)), save=False)
        self._log_param(k, 'max_grad_Lagrangian', max_lagr_grad, save=True)
        self._log_param(k, 'penalty', max_viol)

        return max_viol <= self._tol
