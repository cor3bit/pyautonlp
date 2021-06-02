from typing import Callable, Tuple

import jax.numpy as jnp

from .constants import SolverType
from .fom.gd import GD
from .som.newton import Newton
from .constr.constr_newton import ConstrainedNewton
from .constr.sqp import SQP
from .constr.ip import IP


def solve(
        loss_fn: Callable,
        solver_type: str,
        **kwargs,
) -> Tuple[jnp.ndarray, Tuple]:
    # TODO wrap in try-catch for error handling

    # Redirect to a particular solver based on a solver_type string
    if solver_type == SolverType.GD:
        solver = GD(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.NEWTON:
        if 'eq_constr' in kwargs or 'ineq_constr' in kwargs:
            solver = ConstrainedNewton(loss_fn=loss_fn, **kwargs)
        else:
            solver = Newton(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.SQP:
        solver = SQP(loss_fn=loss_fn, **kwargs)
    elif solver_type == SolverType.IP:
        solver = IP(loss_fn=loss_fn, **kwargs)
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
