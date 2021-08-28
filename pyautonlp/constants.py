class SolverType:
    # First order methods
    GD = 'gd'

    # Second order methods
    NEWTON = 'newton'

    # Constrained
    SQP = 'sqp'
    IP = 'ip'

    # Optimal control methods
    HJB = 'hjb'
    PMP = 'pmp'
    DP = 'dp'
    SINGLE_SHOOTING = 'ss'
    MULT_SHOOTING = 'ms'


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


class KKTForm:
    FULL = 'Full'
    AUGMENTED = 'Augmented'
    NORMAL = 'Normal'

class IntegrateMethod:
    EEULER = 'Explicit Euler'
    RK4 = 'Explicit RK4'
    SSC_EEULER = 'Adaptive Explicit Euler'
    IEULER = 'Implicit Euler'
