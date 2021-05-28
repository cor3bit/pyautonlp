from typing import Callable, Tuple

import jax.numpy as jnp

from .constants import SolverType
from .fom.gd import gd
from .som.newton import newton
from .constr.constr_newton import ConstrainedNewtonSolver
from .constr.sqp import SQP


def solve(
        loss_fn: Callable,
        solver_type: str,
        **kwargs,
) -> Tuple[jnp.ndarray, Tuple]:
    # TODO wrap in try-catch for error handling

    # Redirect to particular solver based on a solver id string
    # solver = None
    if solver_type == SolverType.GD:
        return gd(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.NEWTON:
        if 'eq_constr' in kwargs or 'ineq_constr' in kwargs:
            solver = ConstrainedNewtonSolver(loss_fn=loss_fn, **kwargs)
        else:
            return newton(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.SQP:
        solver = SQP(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.HJB:
        raise NotImplementedError
    elif solver_type == SolverType.PMP:
        raise NotImplementedError
    elif solver_type == SolverType.DP:
        raise NotImplementedError
    elif solver_type == SolverType.DOC:
        raise NotImplementedError
    else:
        raise ValueError(f'Unrecognized solver type: {solver_type}.')

    return solver.solve()
