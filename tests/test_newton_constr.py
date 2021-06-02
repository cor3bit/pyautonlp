import pytest

import jax.numpy as jnp

from pyautonlp.constants import Direction, LineSearch, HessianRegularization
from pyautonlp.constr.constr_newton import ConstrainedNewton


@pytest.fixture()
def setup():
    def loss(x):
        # min z = 0.5 * x_T * x + 1_T * x
        # s.t. x_T * x = 1
        return 0.5 * jnp.dot(x, x) + jnp.sum(x)

    def equality_constr(x):
        # min z = 0.5 * x_T * x + 1_T * x
        # s.t. x_T * x = 1
        return jnp.dot(x, x) - 1

    return loss, equality_constr


def test_newton_exact(setup):
    loss, equality_constr = setup

    solver = ConstrainedNewton(
        loss_fn=loss,
        eq_constr=[equality_constr],
        guess=(-1., 1.),
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_tol=1e-6,
        max_iter=10,
        verbose=False,
    )

    x_star, info = solver.solve()
    converged, loss, k, cache = info

    assert converged
    assert k == 7
    assert -0.9142135 == pytest.approx(loss, abs=1e-6)

    x1, x2 = x_star
    assert -0.70710677 == pytest.approx(x1, abs=1e-6)
    assert -0.70710677 == pytest.approx(x2, abs=1e-6)
