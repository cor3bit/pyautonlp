from typing import Callable

from constants import SolverType


def solve(
        obj_func: Callable,
        solver: str,
        **kwargs,
):
    # TODO redirect to solvers based on solver id
    if solver == SolverType.GD:
        raise NotImplementedError
    elif solver == SolverType.NEWTON:
        raise NotImplementedError
    elif solver == SolverType.HJB:
        raise NotImplementedError
    elif solver == SolverType.PMP:
        raise NotImplementedError
    elif solver == SolverType.DP:
        raise NotImplementedError
    elif solver == SolverType.DOC:
        raise NotImplementedError
    else:
        raise ValueError(f'Unrecognized solver type: {solver}.')
