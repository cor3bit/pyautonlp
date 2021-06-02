import pytest

import jax.numpy as jnp

from pyautonlp.constants import Direction, LineSearch, HessianRegularization
from pyautonlp.constr.sqp import SQP


@pytest.fixture
def setup_qp():
    def loss(x):
        return 0.5 * jnp.dot(x, x)

    def equality_constr(x):
        return jnp.sum(x) - 2

    def inequality_constr1(x):
        return -x[0]

    def inequality_constr2(x):
        return -x[1]

    return loss, equality_constr, inequality_constr1, inequality_constr2


@pytest.fixture
def setup_eq():
    def loss(x):
        return 0.5 * jnp.dot(x, x) + jnp.sum(x)

    def equality_constr(x):
        return jnp.dot(x, x) - 1

    return loss, equality_constr


@pytest.fixture
def setup_ineq():
    def loss(x):
        return 0.5 * jnp.dot(x, x) + jnp.sum(x)

    def eq_constr(x):
        return jnp.dot(x, x) - 1

    def ineq_constr(x):
        return 0.5 - x[0] * x[0] - x[1]

    return loss, eq_constr, ineq_constr


def test_sqp_qp(setup_qp):
    loss, equality_constr, inequality_constr1, inequality_constr2 = setup_qp

    sqp = SQP(
        loss_fn=loss,
        eq_constr=[equality_constr],
        ineq_constr=[inequality_constr1, inequality_constr2],
        guess=(5, 5),
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_tol=1e-6,
        max_iter=5,
        verbose=False,
    )

    x_star, info = sqp.solve()
    converged, loss, k, cache = info

    assert converged
    assert k == 1
    assert 1.0 == pytest.approx(loss)

    x1, x2 = x_star
    assert 1.0 == pytest.approx(x1)
    assert 1.0 == pytest.approx(x2)


def test_sqp_ineq(setup_ineq):
    loss, eq_constr, ineq_constr = setup_ineq

    sqp = SQP(
        loss_fn=loss,
        eq_constr=[eq_constr],
        ineq_constr=[ineq_constr],
        guess=(-1, -1),
        direction=Direction.EXACT_NEWTON,
        reg=HessianRegularization.EIGEN_DELTA,
        line_search=LineSearch.BT_MERIT_ARMIJO,
        beta=0.5,
        gamma=0.1,
        sigma=1.0,
        conv_tol=1e-6,
        max_iter=50,
        verbose=False,
    )

    x_star, info = sqp.solve()
    converged, loss, k, cache = info

    assert not converged
    assert k == 50
    assert -0.79663026 == pytest.approx(loss, abs=1e-6)

    x1, x2 = x_star
    assert -0.93060487508 == pytest.approx(x1, abs=1e-8)
    assert -0.36602541804 == pytest.approx(x2, abs=1e-8)
