import jax.numpy as jnp

import pyautonlp as pan
from pyautonlp.constants import Direction, ConvergenceCriteria, LineSearch, HessianRegularization


def loss(x):
    # min z = 0.5 * x_T * x + 1_T * x
    # s.t. x_T * x = 1
    return 0.5 * jnp.dot(x, x) + jnp.sum(x)


def equality_constr(x):
    # min z = 0.5 * x_T * x + 1_T * x
    # s.t. x_T * x = 1
    return jnp.dot(x, x) - 1


# guess (0., 1.), (-1, -1.),

if __name__ == '__main__':
    sln, info = pan.solve(
        # problem definition
        loss_fn=loss,
        eq_constr=[equality_constr],

        # solver params
        solver_type='newton',
        guess=(0., 1.),
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        alpha=0.1,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_criteria=ConvergenceCriteria.KKT_VIOLATION,
        conv_tol=1e-7,
        max_iter=200,

        # level of details
        verbose=True,
        visualize=True,
        np_loss_fn=lambda x, y: 0.5 * x * x + 0.5 * y * y + x + y,  # used for visualization
        np_eq_constr=[lambda x, y: x * x + y * y - 1],  # used for visualization
    )

    print(sln)
    print(info)
