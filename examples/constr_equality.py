import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import HessianApprox, ConvergenceCriteria, LearningRateStrategy


def loss(x):
    # min z = 0.5 * x_T * x + 1_T * x
    # s.t. x_T * x = 1
    return 0.5 * jnp.dot(x, x) + jnp.sum(x)


def equality_constr(x):
    # min z = 0.5 * x_T * x + 1_T * x
    # s.t. x_T * x = 1
    return jnp.dot(x, x) - 1


if __name__ == '__main__':
    sln, info = pan.solve(
        # problem definition
        loss_fn=loss,
        eq_constr=[equality_constr],

        # solver params
        solver_type='newton',
        guess=(0, 1),
        hessian_approx=HessianApprox.STEEPEST_DESCENT,
        conv_criteria=ConvergenceCriteria.KKT_VIOLATION,
        lr_strategy=LearningRateStrategy.BT_MERIT_ARMIJO,
        lr=0.1,

        # level of details
        verbose=True,
        visualize=True,
    )

    print(sln)
    print(info)
