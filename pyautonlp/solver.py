from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

from pyautonlp.constants import Direction, HessianRegularization


class Solver(ABC):
    _logger = None
    _loss_fn = None
    _x_dims = None
    _cache = None  # info collected during the run

    @abstractmethod
    def solve(
            self,
    ) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    @staticmethod
    def _get_log_str(k, cache_item):
        is_pd = '-' if cache_item.H_pd is None else str(cache_item.H_pd)

        return f'Iteration {k}: Loss: {cache_item.loss:.3f}; ' \
               f'Alpha: {cache_item.alpha:.6f}; ' \
               f'H is pd: {is_pd}; ' \
               f'Sigma: {cache_item.sigma:.2f}; ' \
               f'KKT Penalty: {cache_item.penalty:.5f}.'

    @staticmethod
    def _is_pd_matrix(a: jnp.ndarray) -> bool:
        try:
            L = jnp.linalg.cholesky(a)
            if jnp.isnan(L).any():
                return False
            return True
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
        B_k_is_pd = None

        if (self._direction == Direction.EXACT_NEWTON
                and self._reg != HessianRegularization.NONE):
            B_k_is_pd = True

            Z_k = self._null_space(constr_grad_x.T)

            reduced_B_k = Z_k.T @ B_k @ Z_k

            # check for symmetric
            reduced_B_k = (reduced_B_k + reduced_B_k.T)/2.

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
                B_k += Z_k @ eig_vecs @ jnp.diag(eig_vals_modified-eig_vals) @ eig_vecs.T @ Z_k.T

                # check for symmetric
                B_k = (B_k + B_k.T)/2.

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
