import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization


def loss(x):
    return 0.5 * jnp.dot(x, x) + jnp.sum(x)


def equality_constr(x):
    return jnp.dot(x, x) - 1


def solve_newton(guess):
    return pan.solve(
        # problem definition
        loss_fn=loss,
        eq_constr=[equality_constr],

        # solver params
        solver_type='newton',
        guess=guess,
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        alpha=0.1,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_criteria=ConvergenceCriteria.KKT_VIOLATION,
        conv_tol=1e-6,
        max_iter=100,

        # level of details
        verbose=True,
    )


if __name__ == '__main__':
    pt_names = ['A', 'B', 'C', 'D', 'E']

    initial_guesses = [
        (0., 1.), (1., 1.), (0.5, 1.), (-1., -1.), (-1., 1.),
    ]

    # solve
    caches = []
    for g in initial_guesses:
        sln, info = solve_newton(g)
        print(sln)
        cache = info[-1]
        caches.append(cache)

    # visualize
    for cache, name in zip(caches, pt_names):
        viz = pan.Visualizer(
            loss_fn=lambda x, y: 0.5 * x * x + 0.5 * y * y + x + y,
            eq_constr=[lambda x, y: x * x + y * y - 1],
            solver_caches=[cache],
            cache_names=[name],
            x1_bounds=(-2, 2),
            x2_bounds=(-2, 2),
        )

        viz.plot_convergence()
