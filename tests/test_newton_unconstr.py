import pytest

import jax.numpy as jnp

from pyautonlp.constants import Direction
from pyautonlp.som.newton import Newton


def test_gd_quadratic_exact():
    gd = Newton(
        loss_fn=lambda x: jnp.sum(x * x),
        guess=(5, 5),
        alpha=1.0,
        verbose=False,
    )

    x_star, info = gd.solve()
    converged, loss, k = info

    assert converged
    assert k == 2
    assert 0.0 == pytest.approx(loss, abs=1e-6)

    x1, x2 = x_star
    assert 0.0 == pytest.approx(x1, abs=1e-5)
    assert 0.0 == pytest.approx(x2, abs=1e-5)


@pytest.mark.skip('GN not implemented for Unconstr Newton!')
def test_gd_quadratic_gn():
    gd = Newton(
        loss_fn=lambda x: jnp.sum(x * x),
        guess=(5, 5),
        alpha=0.1,
        direction=Direction.GAUSS_NEWTON,
        verbose=False,
    )

    x_star, info = gd.solve()
    converged, loss, k = info

    assert converged
    assert k == 2
    assert 0.0 == pytest.approx(loss, abs=1e-6)

    x1, x2 = x_star
    assert 0.0 == pytest.approx(x1, abs=1e-5)
    assert 0.0 == pytest.approx(x2, abs=1e-5)
