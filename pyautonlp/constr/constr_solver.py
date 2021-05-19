from typing import Tuple

import jax.numpy as jnp

from pyautonlp.constants import ConvergenceCriteria
from pyautonlp.solver import Solver


class ConstrainedSolver(Solver):
    _constr_fns = None
    _x_dims = None

    def solve(self) -> Tuple[jnp.ndarray, Tuple]:
        raise NotImplementedError

    def _lagrangian(self, x_, multipliers_):
        constr_vals = jnp.array([constraint_fn(x_) for constraint_fn in self._constr_fns], dtype=jnp.float32)
        return self._loss_fn(x_) + jnp.sum(multipliers_ * constr_vals)

    def _lagrangian_full(self, state):
        x = state[:self._x_dims]
        multipliers = state[self._x_dims:]
        constr_vals = jnp.array([constraint_fn(x) for constraint_fn in self._constr_fns], dtype=jnp.float32)
        return self._loss_fn(x) + jnp.sum(multipliers * constr_vals)
