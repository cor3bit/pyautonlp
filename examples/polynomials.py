import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import SolverType


def poly2(x):
    # z = x^2 + y^2
    return jnp.sum(x * x)


if __name__ == '__main__':
    # Gradient Descent solver
    print(f'\nRunning {SolverType.GD} solver.')
    sln, info = pan.solve(
        loss_fn=poly2,
        solver_type='gd',
        guess=(4., 4.),
        learning_rate=0.1,
    )

    print(sln)
    print(info)

    # Newton's Method solver
    print(f'\nRunning {SolverType.NEWTON} solver.')
    sln, info = pan.solve(
        loss_fn=poly2,
        solver_type='newton',
        guess=(4., 4.),
        lr=1.0,
    )

    print(sln)
    print(info)
