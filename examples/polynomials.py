import numpy as np

import pyautonlp as pan


def poly2(x):
    # z = x^2 + y^2
    return x[0] * x[0] + x[1] * x[1]


if __name__ == '__main__':
    res = pan.solve(
        poly2,
        solver='gd',
    )
