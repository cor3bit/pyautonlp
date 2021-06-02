import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import SolverType, Direction


def poly2(x):
    # f(x) = x_1^2 + x_2^2 + ... + x_n^2
    return jnp.sum(x * x)


if __name__ == '__main__':
    # Gradient Descent solver
    print(f'\nRunning {SolverType.GD} solver.')
    sln, info = pan.solve(
        loss_fn=poly2,
        solver_type='gd',
        guess=(4., 4.),
        alpha=0.1,
        verbose=True,
    )

    print(sln)

    # Newton's Method solver
    print(f'\nRunning {SolverType.NEWTON} solver.')
    sln, info = pan.solve(
        loss_fn=poly2,
        solver_type='newton',
        direction=Direction.EXACT_NEWTON,
        guess=(4., 4.),
        alpha=1.0,
        verbose=True,
    )

    print(sln)
