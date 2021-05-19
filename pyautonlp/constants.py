class SolverType:
    # First order methods
    GD = 'gd'

    # Second order methods
    NEWTON = 'newton'

    # Optimal control methods
    HJB = 'hjb'
    PMP = 'pmp'
    DP = 'dp'
    DOC = 'doc'


class HessianApprox:
    EXACT = 'Exact'
    QN = 'Quasi-Newton'
    GN = 'Gauss-Newton'
    BFGS = 'BFGS'


class ConvergenceCriteria:
    KKT_VIOLATION = 'Gradient Norm'
    STEP_DIFF_NORM = 'Step Difference Norm'


class LearningRateStrategy:
    CONST = 'Constant'
    BT = 'Backtracking'
    BT_ARMIJO = 'Backtracking + Armijo'
    BT_WOLFE = 'Backtracking + Wolfe'
