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


class Direction:
    STEEPEST_DESCENT = 'Steepest Descent'
    EXACT_NEWTON = 'Exact'
    GAUSS_NEWTON = 'Gauss-Newton'
    BFGS = 'BFGS'


class LineSearch:
    CONST = 'Constant'
    BT = 'Backtracking'
    BT_ARMIJO = 'Backtracking + Armijo'
    BT_MERIT = 'Backtracking + Merit Fn'
    BT_MERIT_ARMIJO = 'Backtracking + Merit Fn + Armijo'


class HessianRegularization:
    NONE = 'No Regularization'
    EIGEN_DELTA = 'Eigenvalue Modification: delta'
    EIGEN_FLIP = 'Eigenvalue Modification: opposite sign'


class ConvergenceCriteria:
    KKT_VIOLATION = 'Violation of KKT conditions'
    STEP_DIFF_NORM = 'Step Difference Norm'
