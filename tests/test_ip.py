import pytest

import jax.numpy as jnp

from pyautonlp.constants import Direction, LineSearch, HessianRegularization
from pyautonlp.constr.ip import IP


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


def test_ip_qp(setup_qp):
    loss, equality_constr, inequality_constr1, inequality_constr2 = setup_qp

    ip = IP(
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
        verbose=True,
    )

    x_star, info = ip.solve()
    converged, loss, k, cache = info

    assert converged
    assert k == 1
    assert 1.0 == pytest.approx(loss)

    x1, x2 = x_star
    assert 1.0 == pytest.approx(x1)
    assert 1.0 == pytest.approx(x2)
