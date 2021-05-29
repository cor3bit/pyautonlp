import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization


def loss(x):
    return 0.5 * jnp.dot(x, x)


def equality_constr(x):
    return jnp.sum(x) - 2


def inequality_constr1(x):
    return -x[0]


def inequality_constr2(x):
    return -x[1]


if __name__ == '__main__':
    sln, info = pan.solve(
        # problem definition
        loss_fn=loss,
        eq_constr=[equality_constr],
        ineq_constr=[inequality_constr1, inequality_constr2],

        # solver params
        solver_type='sqp',
        guess=(4, 4),
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT,
        alpha=0.1,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_criteria=ConvergenceCriteria.KKT_VIOLATION,
        conv_tol=1e-6,
        max_iter=20,

        # level of details
        verbose=True,
    )

    print(sln)
