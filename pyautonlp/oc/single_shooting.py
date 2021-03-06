import logging
from time import perf_counter
from typing import List, Tuple, Callable, Optional, Union
from functools import partial

import jax.numpy as jnp
from jax import grad, jacfwd, jit

from pyautonlp.constants import *
from pyautonlp.viz import Visualizer
from pyautonlp.constr.constr_solver import ConstrainedSolver
from pyautonlp.oc.integrators import integrate, integrate_with_ad


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
        self._logger = logging.getLogger('single_shooting')
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing solver.')

        # oc params
        self._dynamics = jit(dynamics)
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

        self._n_w = self._n_steps * self._n_u
        self._logger.info(f'Dimensions of the decision vector w: {self._n_w}.')
        self._w_u_max = jnp.full(shape=(self._n_w,), fill_value=self._u_max, dtype=jnp.float32)
        self._w_u_min = jnp.full(shape=(self._n_w,), fill_value=self._u_min, dtype=jnp.float32)

        # residuals R(w)
        self._n_r = self._n_steps * (self._n_x + self._n_u) + self._n_x
        self._logger.info(f'Dimensions of the residual vector R(w): {self._n_r}.')

        # constraints
        # equality constraints on the last state x(N)
        self._n_eq_mult = self._n_x
        self._logger.info(f'Dimensions of the equality multiplier vector g(w): {self._n_eq_mult}.')

        # inequality constraint (u_max and u_min) for each control variable
        self._n_ineq_mult = 2 * self._n_w
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

        # control variable, flatten array of u_0, u_1, ..., u_{N-1}
        w_k = jnp.zeros((self._n_w,), dtype=jnp.float32)
        # w_k = jnp.full((self._w_dims,), fill_value=50., dtype=jnp.float32)

        # lagrange multipliers
        m_eq_k = jnp.zeros((self._n_eq_mult,), dtype=jnp.float32)
        m_ineq_k = jnp.zeros((self._n_ineq_mult,), dtype=jnp.float32)

        # constraints preprocessing
        grad_c_ineq_k = jnp.empty((self._n_ineq_mult, self._n_w), jnp.float32)
        grad_c_ineq_k = grad_c_ineq_k.at[::2, :].set(
            jnp.diag(jnp.ones((self._n_w,), jnp.float32))
        )
        grad_c_ineq_k = grad_c_ineq_k.at[1::2, :].set(
            jnp.diag(-jnp.ones((self._n_w,), jnp.float32))
        )

        # precompiles dynamics jacobians
        dfds_fn = jit(jacfwd(self._dynamics, argnums=0))
        dfdu_fn = jit(jacfwd(self._dynamics, argnums=1))

        # create control discretization grid
        tg_shooting = jnp.linspace(self._t0, self._tf, num=self._n_steps + 1)

        # pure loss function for backtracking
        self._loss_fn = partial(self._ss_loss_fn, k=0, time_grid_shooting=tg_shooting,
                                dfds_fn=dfds_fn, dfdu_fn=dfdu_fn, with_grads=False)

        # loss penalties
        sqrt_x_pen = jnp.sqrt(self._x_pen)
        # sqrt_u_pen = jnp.sqrt(self._u_pen.reshape((-1,)))

        # run main SQP loop
        converged = False
        k = 0

        # log initial parameters
        self._log_param(k, 'n_steps', self._n_steps, display=False)
        self._log_param(k, 'x0', self._x0, display=False)
        self._log_param(k, 'xf', self._xf, display=False)

        while k < self._max_iter:
            self._logger.info(f'---------------- Iteration {k} ----------------')
            self._log_param(k, 'u', w_k, display=False)
            # self._log_param(k, 'm_eq', m_eq_k)
            # self._log_param(k, 'm_ineq', m_ineq_k)

            # given w_k, calculate evolution of x
            loss, x_i, grad_loss, B_k, Jac_R = self._ss_loss_fn(
                w_k, k, tg_shooting, dfds_fn, dfdu_fn, with_grads=True)

            # construct constraints matrixes
            c_eq_k = self._eval_eq_constraints(x=w_k, x_i=x_i)
            c_ineq_k = self._eval_ineq_constraints(x=w_k)

            # undo *sqrt(Q) for pure dx(n)/dw
            a = Jac_R[-self._n_x:, :]
            b = jnp.repeat(sqrt_x_pen, self._n_w).reshape((self._n_x, self._n_w))
            grad_c_eq_k = a / b

            converged = self._ss_converged(k, grad_loss, m_eq_k, c_eq_k, grad_c_eq_k,
                                           m_ineq_k, c_ineq_k, grad_c_ineq_k)

            if converged:
                break

            # find direction by solving QP
            # c_k = jnp.concatenate([c_eq_k, c_ineq_k])
            # grad_c_k = jnp.vstack([grad_c_eq_k, grad_c_ineq_k])

            # TODO REMOVE, DEBUG-ONLY
            # np_Jac_R = np.array(Jac_R)
            # np_B_k = np.array(B_k)
            # np_grad_loss = np.array(grad_loss)
            # np_c_k = np.array(c_k)
            # np_grad_c_k = np.array(grad_c_k)
            # np_b = np.array(b)
            # np_a = np.array(a)

            # TODO relax if infeasible
            try:
                # d_k = self._solve_qp(B_k, grad_loss, c_k, grad_c_k.T, self._n_eq_mult)
                d_k = self._solve_qp_cvxopt(B_k, grad_loss, c_eq_k, c_ineq_k,
                                            grad_c_eq_k, grad_c_ineq_k)
            except Exception as e:
                self._logger.warning(f'QP is infeasible! Failed with {e}.')

                self._logger.info('Trying unbounded solver.')
                d_k = self._solve_qp(B_k, grad_loss, c_eq_k, grad_c_eq_k.T, self._n_eq_mult)
                self._log_param(k, 'max_d', jnp.max(d_k), save=False)
                self._log_param(k, 'min_d', jnp.min(d_k), save=False)

                self._logger.info('Trying infeasibility minimization.')
                d_k = self._solve_infeasible_qp(B_k, grad_loss, c_eq_k, c_ineq_k,
                                                grad_c_eq_k, grad_c_ineq_k)

            self._log_param(k, 'max_d', jnp.max(d_k), save=False)
            self._log_param(k, 'min_d', jnp.min(d_k), save=False)

            # TODO expiremental sigma
            # self._adjust_sigma(grad_loss, d_k, c_eq_k, B_k)
            self._log_param(k, 'sigma', self._sigma)

            # calculate step size (line search)
            alpha_k = self._ss_backtrack(w_k=w_k, d_k=d_k, loss=loss, grad_loss=grad_loss,
                                         x_i=x_i, max_iter=10)

            # self._log_param(k, 'sigma', self._sigma)
            self._log_param(k, 'alpha', alpha_k)

            # update controls and multipliers
            w_k += alpha_k * d_k[:self._n_w]

            m_eq_ind_end = self._n_w + self._n_eq_mult
            m_eq_k = (1 - alpha_k) * m_eq_k + alpha_k * d_k[self._n_w:m_eq_ind_end]
            m_ineq_k = (1 - alpha_k) * m_ineq_k + alpha_k * d_k[m_eq_ind_end:]

            self._sigma = jnp.max(jnp.abs(jnp.concatenate([m_eq_k, m_ineq_k]))) + 0.1
            # self._sigma = jnp.max(jnp.abs(m_eq_k)) + 0.1

            # increment counter
            k += 1

        if k < self._max_iter:
            self._logger.info(f'Converged after {k} iterations!')
        else:
            self._logger.info(f'Maximum number of {k} iterations reached.')

        # fill additional info
        info = (converged, k, self._cache)

        # charts
        if self._viz:
            Visualizer(
                solver_type=SolverType.SINGLE_SHOOTING,
                solver_caches=[self._cache],
                save_dir=self._save_plot_dir,
                x1_bounds=(self._u_min, self._u_max),
                n=self._visualize_n,
            ).plot_shooting()

        self._logger.info(f'Total solving time: {(perf_counter() - solve_t0) / 60.:.3f} min.')

        return w_k, info

    def _ss_loss_fn(
            self,
            w_k: jnp.ndarray,
            k: int,
            time_grid_shooting: jnp.ndarray,
            dfds_fn: Callable,
            dfdu_fn: Callable,
            with_grads: bool = False,
    ) -> Union[Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
               Tuple[float, jnp.ndarray]]:
        # logging
        if with_grads:
            self._logger.info('Loss calculation started.')

        t1 = perf_counter()

        # initial params
        x_i = self._x0
        R = []
        Jac_R = jnp.zeros((self._n_r, self._n_w), jnp.float32) if with_grads else None
        dxdw_prev = jnp.zeros((self._n_x, self._n_w), jnp.float32) if with_grads else None
        sqrt_x_pen = jnp.sqrt(self._x_pen)
        sqrt_x_pen_array = jnp.repeat(sqrt_x_pen, self._n_w).reshape((self._n_x, self._n_w))
        sqrt_u_pen = jnp.sqrt(self._u_pen.reshape((-1,)))

        # convert w -> u
        # Note. w is flat, u is (n_steps-1)x(n_u)
        u_k = jnp.reshape(w_k, (self._n_steps, self._n_u))

        for i, (t_start, t_end, u_i) in enumerate(zip(time_grid_shooting[:-1], time_grid_shooting[1:], u_k)):
            time_grid_integration = jnp.linspace(t_start, t_end, num=self._n_steps_internal + 1)

            x_full, info = integrate(self._dynamics, x_i, u_i, time_grid_integration,
                                     method=IntegrateMethod.RK4, with_grads=True,
                                     dfds_fn=dfds_fn, dfdu_fn=dfdu_fn)
            x_j = x_full[-1]

            # DEBUG-ONLY: Check sensitivities with AD
            # x1, G_x1, G_u1 = integrate_with_ad(self._dynamics, x_i, u_i, time_grid_integration,
            #                          method=IntegrateMethod.RK4)
            # a = np.array(G_x-G_x1)
            # b = np.array(G_u-G_u1.reshape((-1)))
            # c = np.array(G_x.T-G_x1)

            # R components: calc loss based on the starting values
            loss_x_i = (x_i - self._xf) * sqrt_x_pen
            loss_u_i = u_i * sqrt_u_pen
            R.append(loss_x_i)
            R.append(loss_u_i)

            if with_grads:
                G_x = info['G_x']
                G_u = info['G_u'].reshape((-1))

                # Jac_R components: chain rule, one row at a time
                diag_component = G_u
                dxdw_next = G_x @ dxdw_prev
                dxdw_next = dxdw_next.at[:, i].set(diag_component)

                start_ind = (i + 1) * (self._n_x + self._n_u)
                end_ind = start_ind + self._n_x
                Jac_R = Jac_R.at[start_ind:end_ind, :].set(dxdw_next * sqrt_x_pen_array)

                # fill du_k/du_k
                Jac_R = Jac_R.at[start_ind - 1:start_ind - 1 + self._n_u, i].set(sqrt_u_pen)

                # update dxdw
                dxdw_prev = dxdw_next

            # update x_i
            x_i = x_j

        # last time step - x(N)
        loss_x_i = (x_i - self._xf) * sqrt_x_pen
        R.append(loss_x_i)

        # convert to one vector
        R = jnp.concatenate(R)
        assert R.shape[0] == self._n_r

        loss = 0.5 * R.T @ R

        grad_loss = Jac_R.T @ R if with_grads else None
        B_k = Jac_R.T @ Jac_R if with_grads else None

        # logging
        if with_grads:
            self._log_param(k, 'x_N', x_i, save=False)
            self._log_param(k, 'loss', loss, save=with_grads)
            self._logger.info(f'Loss calculation finished in {perf_counter() - t1:.3f} sec.')

        return (loss, x_i, grad_loss, B_k, Jac_R) if with_grads else (loss, x_i)

    def _eval_eq_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        return kwargs['x_i'] - self._xf

    def _eval_ineq_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        c_ineq_k = jnp.empty((2 * self._n_w,), jnp.float32)
        c_ineq_k = c_ineq_k.at[::2].set(x - self._w_u_max)
        c_ineq_k = c_ineq_k.at[1::2].set(-x + self._w_u_min)
        return c_ineq_k

    def _eval_constraints(
            self,
            x: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        c_eq_k = self._eval_eq_constraints(x, **kwargs)
        c_ineq_k = self._eval_ineq_constraints(x, **kwargs)
        return jnp.concatenate([c_eq_k, c_ineq_k])

    def _adjust_sigma(
            self,
            grad_loss: jnp.ndarray,
            d_k: jnp.ndarray,
            c_eq_k: jnp.ndarray,
            B_k: jnp.ndarray,
            rho: float = 0.1,
            eps: float = 0.01,
    ):
        d_w = d_k[:self._n_w]
        s_num = jnp.dot(grad_loss, d_w) + 0.5 * d_w.T @ B_k @ d_w
        s_den = (1 - rho) * jnp.linalg.norm(c_eq_k, ord=1)
        sigma_barrier = s_num / s_den
        sigma = sigma_barrier + eps
        # self._sigma = jnp.maximum(1., sigma)
        # if sigma < 0:
        #     sigma = eps

        sigma = jnp.maximum(1., sigma)

        self._sigma = sigma

    def _ss_backtrack(
            self,
            w_k: jnp.ndarray,
            d_k: jnp.ndarray,
            loss: float,
            grad_loss: jnp.ndarray,
            x_i: jnp.ndarray,
            initial_alpha: float = 1.,
            max_iter: int = 30,
            **kwargs
    ) -> float:
        self._logger.info('Backtracking started.')

        alpha = initial_alpha

        direction_w = d_k[:self._n_w]

        curr_loss = loss
        curr_merit_adj = self._ss_merit_adj(w_k, x_i)
        armijo_adj = self._ss_armijo_adj(alpha, direction_w, grad_loss, curr_merit_adj)

        # self._log_param(0, 'L_k', curr_loss, save=False)
        # self._log_param(0, 'M_k', curr_merit_adj, save=False)
        # self._log_param(0, 'A_k', armijo_adj, save=False)

        next_w_k = w_k + alpha * direction_w
        next_loss, next_x_i = self._loss_fn(next_w_k)
        next_merit_adj = self._ss_merit_adj(next_w_k, next_x_i)

        # self._log_param(0, 'L_{k+1}', next_loss, save=False)
        # self._log_param(0, 'M_{k+1}', next_merit_adj, save=False)

        n_iter = 0
        while (next_loss + next_merit_adj >= curr_loss + curr_merit_adj + armijo_adj) and (n_iter < max_iter):
            # update alpha
            alpha *= self._beta

            # update all alpha dependencies
            armijo_adj *= self._beta

            next_w_k = w_k + alpha * direction_w
            next_loss, next_x_i = self._loss_fn(next_w_k)
            next_merit_adj = self._ss_merit_adj(next_w_k, next_x_i)

            # self._log_param(0, 'A_k', armijo_adj, save=False)
            # self._log_param(0, 'L_{k+1}', next_loss, save=False)
            # self._log_param(0, 'M_{k+1}', next_merit_adj, save=False)

            n_iter += 1

        if n_iter == max_iter:
            self._logger.warning(f'Backtracking failed to find alpha after {max_iter} iterations!')
        else:
            self._logger.info(f'Backtracking converged after {n_iter} iterations.')

        return alpha

    def _ss_merit_adj(
            self,
            w_k: jnp.ndarray,
            x_i: jnp.ndarray,
    ) -> float:
        c_eq_k = self._eval_eq_constraints(x=w_k, x_i=x_i)
        eq_norm = jnp.linalg.norm(c_eq_k, ord=1)

        # ineq_norm = 0.
        # TODO skips inequality adjustment

        c_ineq_k = self._eval_ineq_constraints(x=w_k)
        ineq_norm = jnp.linalg.norm(jnp.clip(c_ineq_k, a_min=0), ord=1)

        return self._sigma * (eq_norm + ineq_norm)

    def _ss_armijo_adj(
            self,
            alpha: float,
            direction_w: jnp.ndarray,
            grad_loss: jnp.ndarray,
            merit_adj: float,
    ) -> float:
        return self._gamma * alpha * (jnp.dot(grad_loss, direction_w) - merit_adj)

    def _ss_converged(
            self,
            k: int,
            grad_loss: jnp.ndarray,
            m_eq_k: jnp.ndarray,
            c_eq_k: jnp.ndarray,
            grad_c_eq_k: jnp.ndarray,
            m_ineq_k: jnp.ndarray,
            c_ineq_k: jnp.ndarray,
            grad_c_ineq_k: jnp.ndarray,
    ) -> bool:
        max_c_eq = jnp.max(jnp.abs(c_eq_k))
        max_c_ineq = jnp.max(jnp.clip(c_ineq_k, a_min=0))
        grad_lagr = grad_loss + m_eq_k @ grad_c_eq_k + m_ineq_k @ grad_c_ineq_k

        # self._logger.info(f'm_eq_k: {m_eq_k}')
        # self._logger.info(f'm_ineq_k: {m_ineq_k}')
        # self._logger.info(f'grad_c_eq_k: {grad_c_eq_k}')
        # self._logger.info(f'grad_c_ineq_k: {grad_c_ineq_k}')

        max_lagr_grad = jnp.max(jnp.abs(grad_lagr))
        max_viol = jnp.maximum(jnp.maximum(max_c_eq, max_c_ineq), max_lagr_grad)

        self._log_param(k, 'max_c_eq', max_c_eq)
        self._log_param(k, 'max_c_ineq', max_c_ineq)
        # self._log_param(k, 'max_grad_loss', jnp.max(jnp.abs(grad_loss)), save=False)
        self._log_param(k, 'max_grad_Lagrangian', max_lagr_grad)
        self._log_param(k, 'penalty', max_viol)

        return max_viol <= self._tol
