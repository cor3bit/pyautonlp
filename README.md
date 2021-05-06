## PyAutoNLP - Solving NLP problems with PyTorch autodiff backend


### Usage

- Install the requirements with `pip`:

```
$ pip install -r requirements.txt
```

- Specify the problem:

TODO

- Run the solver:

TODO


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