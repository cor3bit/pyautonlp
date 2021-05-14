from jax import jit, jacfwd, jacrev


def hessian(func):
    return jit(jacfwd(jacrev(func)))
