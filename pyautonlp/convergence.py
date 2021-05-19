import jax.numpy as jnp


def kkt_violation(grad_at_point, tol):
    inf_norm = jnp.max(jnp.abs(grad_at_point))
    return inf_norm < tol


def step_difference():
    raise NotImplementedError
