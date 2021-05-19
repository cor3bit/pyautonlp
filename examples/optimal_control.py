import jax.numpy as jnp

import pyautonlp as pan

if __name__ == '__main__':
    sln, info = pan.solve(
        oc,
        solver_type='pmp',
        guess=(4., 4.),
        learning_rate=0.1,
    )

    print(sln)
    print(info)
