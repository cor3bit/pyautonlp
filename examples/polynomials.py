import jax.numpy as jnp

import pyautonlp as pan


def poly2(x):
    # z = x^2 + y^2
    return jnp.sum(x * x)


if __name__ == '__main__':
    res = pan.solve(
        poly2,
        solver='gd',
        guess=(4., 4.),
        learning_rate=0.1,
    )

    print(res)
