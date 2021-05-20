import pytest

import jax.numpy as jnp


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


def test_newton_equality1(setup):
    loss, equality_constr = setup

    assert 1 == 1
