## PyAutoNLP - JAX Implementation of Differentiable NLP Solvers

### Usage

- Install the library with `pip`:

```
$ pip install git+https://github.com/cor3bit/pyautonlp.git
```

- Specify the problem using `jax.numpy` arrays:

```python
import jax.numpy as jnp

def loss(x):
    return 0.5 * jnp.dot(x, x) + jnp.sum(x)

def equality_constr(x):
    return jnp.dot(x, x) - 1
```

- Select a solver and run PyAutoNLP `solve()` method:

```python
import pyautonlp as pan

sln, info = pan.solve(
    loss_fn=loss,
    eq_constr=[equality_constr],
    solver_type='newton',
    guess=(1., 1.),
)
```

### Supported Solvers

Currently supported methods:

- First Order Methods
    - Gradient Descent (solver id: 'gd')
- Second Order Methods
    - Newton's method (solver id: 'newton')
- Constrained Optimization
    - Newton's method (solver id: 'newton')
    - SQP (solver id: 'sqp')
    - IP (solver id: 'ip')
- Optimal Control [in progress]
    - HJB (solver id: 'hjb')
    - Pontryagin's method (solver id: 'pmp')
    - Dynamic Programming (solver id: 'dp')
    - Direct Optimal Control (solver id: 'doc')

### Acknowledgements

- The library was developed following Mario Zanon's lectures on Numerical Methods for Optimal Control.
  Web: https://mariozanon.wordpress.com/numerical-methods-for-optimal-control/

- The optimization algorithms are based on JAX auto-diff toolkit. Web: https://github.com/google/jax

- The visualization of the algorithm performance is inspired by Jaewan Yun's library
  (https://github.com/Jaewan-Yun/optimizer-visualization)
  and Louis Tiao's post
  (http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/)

### References

More detailed description of the optimization algorithms can be found in

1. Numerical Optimization - Jorge Nocedal and Stephen J. Wright, Springer, 2006.
2. Algorithms for Optimization - Mykel J. Kochenderfer and Tim A. Wheeler, MIT Press, 2019.
3. Optimal Control Theory - Suresh P. Sethi, Springer, 2019.