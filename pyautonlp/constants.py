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
    GAUSS_NEWTON = 'Gauss-Newton'
    BFGS = 'BFGS'
    STEEPEST_DESCENT = 'Steepest Descent'


class ConvergenceCriteria:
    KKT_VIOLATION = 'Gradient Norm'
    STEP_DIFF_NORM = 'Step Difference Norm'


class LearningRateStrategy:
    CONST = 'Constant'
    BT = 'Backtracking'
    BT_ARMIJO = 'Backtracking + Armijo'
    BT_MERIT = 'Backtracking + Merit Fn'
    BT_MERIT_ARMIJO = 'Backtracking + Merit Fn + Armijo'
