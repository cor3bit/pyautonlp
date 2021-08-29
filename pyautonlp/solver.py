from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

from pyautonlp.constants import Direction, HessianRegularization


class Solver(ABC):
    _logger = None
    _loss_fn = None
    _x_dims = None
    _step_cache = None  # info collected during the step
    _cache = None  # full info collected during the run

    @abstractmethod
    def solve(
            self,
    ) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    def _log_step(self):
        assert self._step_cache is not None
        assert 'k' in self._step_cache
        iter_i = self._step_cache['k']
        dict_str = ','.join([f' {key}: {v:.6f}' for key, v in self._step_cache.items()
                             if key not in ('k', 'x') and v is not None])
        msg = f'Iteration {iter_i}:' + dict_str
        self._logger.info(msg)

    def _log_param(self, k, name, value, save=True):
        # save in cache
        if save:
            if k not in self._cache:
                self._cache[k] = {}
            self._cache[k][name] = value

        # display message
        msg = f' {name}: {value}'
        self._logger.info(msg)

    # @staticmethod
    def _is_pd_matrix(self, a: jnp.ndarray) -> bool:
        try:
            # check that A is symmetric
            if jnp.allclose(a, a.T, rtol=1e-5, atol=1e-8):
                L = jnp.linalg.cholesky(a)
                if jnp.isnan(L).any():
                    return False
                return True
            else:
                self._logger.warning(f'Hessian is not symmetric!')
                return False

        except Exception as e:
            return False

    def _hessian_exact(self, fn: Callable) -> Callable:
        return jit(jacfwd(jacrev(fn)))

    def _hessian_bfgs_approx(self, B_prev, g_k, g_prev, x_k, x_prev, **kwargs) -> jnp.ndarray:
        if B_prev is None:
            return jnp.eye(N=self._x_dims)  # B_0
        else:
            y = x_k - x_prev
            s = g_k - g_prev

            c1 = jnp.inner(y, s)
            incr1 = jnp.outer(y, y) / c1

            c2 = s @ B_prev @ s
            incr2 = B_prev @ jnp.outer(s, s) @ B_prev / c2

            # TODO check BFGS
            if jnp.isclose(c1, 0) or jnp.isclose(c2, 0):
                self._logger.warning(f'BFGS: constants are zero.')
                return B_prev

            B_k = B_prev + incr1 - incr2

            return B_k

    def _hessian_gn_approx(self, g_k, **kwargs) -> jnp.ndarray:
        return jnp.outer(g_k, g_k)

    def _hessian_sd_approx(self, **kwargs) -> jnp.ndarray:
        return jnp.eye(N=self._x_dims)

    def _regularize_hessian(self, B_k, constr_grad_x):
        B_k_is_pd = True

        if (self._direction == Direction.EXACT_NEWTON and self._reg != HessianRegularization.NONE):
            # B_k_is_pd = self._is_pd_matrix(B_k)
            # if not B_k_is_pd:

            Z_k = self._null_space(constr_grad_x.T)

            reduced_B_k = Z_k.T @ B_k @ Z_k

            # check for symmetric
            # reduced_B_k = (reduced_B_k + reduced_B_k.T) / 2.

            eig_vals, eig_vecs = jnp.linalg.eigh(reduced_B_k)
            delta = 1e-6
            min_eig = jnp.min(eig_vals)

            if min_eig < delta:
                B_k_is_pd = False
                if self._reg == HessianRegularization.EIGEN_DELTA:
                    eig_vals_modified = eig_vals.at[eig_vals < delta].set(delta)
                elif self._reg == HessianRegularization.EIGEN_FLIP:
                    eig_vals_modified = jnp.array([self._flip_eig(e, delta) for e in eig_vals])
                else:
                    # TODO modified Cholesky
                    raise NotImplementedError

                # TODO diag, just modified not -orig
                diag_eig = jnp.diag(eig_vals_modified - eig_vals)
                delta_B = Z_k @ eig_vecs @ diag_eig @ eig_vecs.T @ Z_k.T

                # eig_vals3, eig_vecs3 = jnp.linalg.eigh(delta_B)

                B_k += delta_B

                # check for symmetric
                # B_k = (B_k + B_k.T) / 2.

                # Debug
                # eig_vals2, eig_vecs2 = jnp.linalg.eigh(B_k)
                # is_pd = self._is_pd_matrix(B_k)

        # ensure symmetric
        assert jnp.allclose(B_k, B_k.T, rtol=1e-5, atol=1e-8)

        return B_k, B_k_is_pd

    def _regularize_full_hessian(self, B_k):
        B_k_is_pd = True

        if (self._direction == Direction.EXACT_NEWTON and self._reg != HessianRegularization.NONE):
            # B_k_is_pd = self._is_pd_matrix(B_k)
            # if not B_k_is_pd:

            # Z_k = self._null_space(constr_grad_x.T)
            #
            # reduced_B_k = Z_k.T @ B_k @ Z_k

            # check for symmetric
            # reduced_B_k = (reduced_B_k + reduced_B_k.T) / 2.

            eig_vals, eig_vecs = jnp.linalg.eigh(B_k)
            delta = 1e-6
            min_eig = jnp.min(eig_vals)

            if min_eig < delta:
                B_k_is_pd = False
                if self._reg == HessianRegularization.EIGEN_DELTA:
                    eig_vals_modified = eig_vals.at[eig_vals < delta].set(delta)
                elif self._reg == HessianRegularization.EIGEN_FLIP:
                    eig_vals_modified = jnp.array([self._flip_eig(e, delta) for e in eig_vals])
                else:
                    # TODO modified Cholesky
                    raise NotImplementedError

                # TODO diag, just modified not -orig
                diag_eig = jnp.diag(eig_vals_modified)
                # delta_B = eig_vecs @ diag_eig @ eig_vecs.T

                # eig_vals3, eig_vecs3 = jnp.linalg.eigh(delta_B)

                B_k = eig_vecs @ diag_eig @ eig_vecs.T

                # check for symmetric
                # B_k = (B_k + B_k.T) / 2.

                # Debug
                # eig_vals2, eig_vecs2 = jnp.linalg.eigh(B_k)
                # is_pd = self._is_pd_matrix(B_k)

        # ensure symmetric
        # assert jnp.allclose(B_k, B_k.T, rtol=1e-5, atol=1e-8)

        return B_k, B_k_is_pd

    @staticmethod
    def _null_space(a: jnp.ndarray, rcond=None):
        # mimics numpy null_space
        u, s, vh = jnp.linalg.svd(a, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]

        if rcond is None:
            rcond = jnp.finfo(s.dtype).eps * max(M, N)

        tol = jnp.amax(s) * rcond
        num = jnp.sum(s > tol, dtype=int)
        Q = vh[num:, :].T.conj()
        return Q

    @staticmethod
    def _flip_eig(e, delta):
        if jnp.isclose(e, 0.):
            return delta
        elif e < 0:
            return -e
        else:
            return e
