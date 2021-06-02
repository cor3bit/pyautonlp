import pytest

import jax.numpy as jnp

from pyautonlp.fom.gd import GD


def test_gd_quadratic():
    gd = GD(
        loss_fn=lambda x: jnp.sum(x * x),
        guess=(5, 5),
        alpha=0.1,
        verbose=False,
    )

    x_star, info = gd.solve()
    converged, loss, k = info

    assert converged
    assert k == 67
    assert 0.0 == pytest.approx(loss, abs=1e-6)

    x1, x2 = x_star
    assert 0.0 == pytest.approx(x1, abs=1e-5)
    assert 0.0 == pytest.approx(x2, abs=1e-5)
