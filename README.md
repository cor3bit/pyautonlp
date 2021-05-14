## PyAutoNLP - Solving NLP problems with JAX autodiff backend


### Usage

- Install the requirements with `pip`:

```
$ pip install -r requirements.txt
```

- Specify the problem using `jax.numpy` arrays:

```python
import jax.numpy as jnp

def poly2(x):
    # z = x^2 + y^2
    return jnp.sum(x * x)
```

- Select a solver and run PyAutoNLP `solve()` method:

```python
import pyautonlp as pan

sln, info = pan.solve(
    poly2,
    solver='newton',
    guess=(4., 4.),
    learning_rate=0.1,
)
```


### Supported Solvers

Currently the following methods are supported:
- First Order Methods
    - Gradient Descent (solver id: 'gd')
- Second Order Methods
    - Newton's method (solver id: 'newton')
- Optimal Control Methods
    - HJB (solver id: 'hjb')
    - Pontryagin's method (solver id: 'pmp') 
    - Dynamic Programming (solver id: 'dp')
    - Direct Optimal Control (solver id: 'doc')


### References

1. Numerical Optimization - Jorge Nocedal 
   and Stephen J. Wright, Springer, 2006.
2. Algorithms for Optimization - Mykel J. Kochenderfer and 
   Tim A. Wheeler, MIT Press, 2019.
3. Optimal Control Theory - Suresh P. Sethi, 
Springer, 2019.
4. JAX: Autograd and XLA. Web: https://github.com/google/jax
