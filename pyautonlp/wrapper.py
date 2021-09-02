from typing import Callable, Tuple

import jax.numpy as jnp

from .constants import SolverType
from .fom.gd import GD
from .som.newton import Newton
from .constr.constr_newton import ConstrainedNewton
from .constr.sqp import SQP
from .constr.ip import IP
from .oc.single_shooting import SingleShooting
from .oc.multiple_shooting import MultipleShooting


def solve(
        solver_type: str,
        **kwargs,
) -> Tuple[jnp.ndarray, Tuple]:
    # TODO wrap in try-catch for error handling

    # Redirect to a particular solver based on a solver_type string
    if solver_type == SolverType.GD:
        solver = GD(**kwargs)
    elif solver_type == SolverType.NEWTON:
        if 'eq_constr' in kwargs or 'ineq_constr' in kwargs:
            solver = ConstrainedNewton(**kwargs)
        else:
            solver = Newton(**kwargs)
    elif solver_type == SolverType.SQP:
        solver = SQP(**kwargs)
    elif solver_type == SolverType.IP:
        solver = IP(**kwargs)
    elif solver_type == SolverType.SINGLE_SHOOTING:
        solver = SingleShooting(**kwargs)
    elif solver_type == SolverType.MULT_SHOOTING:
        solver = MultipleShooting(**kwargs)
    elif solver_type == SolverType.HJB:
        raise NotImplementedError
    elif solver_type == SolverType.PMP:
        raise NotImplementedError
    elif solver_type == SolverType.DP:
        raise NotImplementedError
    else:
        raise ValueError(f'Unrecognized solver type: {solver_type}.')

    return solver.solve()
