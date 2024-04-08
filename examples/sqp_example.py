import string

import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import (SolverType, Direction, ConvergenceCriteria,
                                 LineSearch, HessianRegularization)

VERBOSE = True
MAX_ITER = 100


def solve_with_pan(solver_type, initial_guess, loss_fn, eq_constr, ineq_constr=None):
    return pan.solve(
        # problem definition
        loss_fn=loss_fn,
        eq_constr=eq_constr,
        ineq_constr=ineq_constr,

        # solver params
        solver_type=solver_type,
        guess=initial_guess,
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_criteria=ConvergenceCriteria.KKT_VIOLATION,
        conv_tol=1e-6,
        max_iter=MAX_ITER,

        # level of details
        verbose=VERBOSE,
    )


def example1():
    def example1_loss(x):
        # min z = 0.5 * x_T * x + 1_T * x
        # s.t. x_T * x = 1
        return 0.5 * jnp.dot(x, x) + jnp.sum(x)

    def example1_constr(x):
        # min z = 0.5 * x_T * x + 1_T * x
        # s.t. x_T * x = 1
        return jnp.dot(x, x) - 1

    eq_cons = [example1_constr]

    initial_guesses = [
        (0., 1.),
        (-1., -1.),
        # (-1., 1.),
        # (0., 0.),
        # (1., 1.),
        # (1., 1. + 1e-6),
        # (0.5, 1.),
    ]

    pt_names = string.ascii_uppercase[:len(initial_guesses)]

    # solve
    caches = []
    for g in initial_guesses:
        sln, info = solve_with_pan(SolverType.SQP, g, example1_loss, eq_cons)
        print(sln)
        # print(info)
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


def example2():
    def example2_loss(x):
        return 0.5 * jnp.dot(x, x) + jnp.sum(x)

    def example2_eq_constr(x):
        return jnp.dot(x, x) - 1

    def example2_ineq_constr(x):
        return 0.5 - x[0] * x[0] - x[1]

    eq_cons = [example2_eq_constr]

    ineq_cons = [example2_ineq_constr]

    initial_guesses = [
        (0, 1),
        (-1, -1),
        # (0.9, 1),
        # (1, -1),
    ]

    pt_names = string.ascii_uppercase[:len(initial_guesses)]

    # solve
    caches = []
    for g in initial_guesses:
        sln, info = solve_with_pan(SolverType.SQP, g, example2_loss, eq_cons, ineq_cons)
        print(sln)
        # print(info)
        cache = info[-1]
        caches.append(cache)

    # # visualize
    for cache, name in zip(caches, pt_names):
        viz = pan.Visualizer(
            loss_fn=lambda x, y: 0.5 * x * x + 0.5 * y * y + x + y,
            eq_constr=[lambda x, y: x * x + y * y - 1],
            ineq_constr=[lambda x, y: 0.5 - x * x - y],
            solver_caches=[cache],
            cache_names=[name],
            x1_bounds=(-2, 2),
            x2_bounds=(-2, 2),
        )

        viz.plot_convergence()


# -------------- Runner --------------

if __name__ == '__main__':
    example1()
    example2()
