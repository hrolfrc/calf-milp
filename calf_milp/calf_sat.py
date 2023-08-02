"""
This is calf-sat, an implementation of CALF using MILP.

"""

import numpy as np
from ortools.linear_solver import pywraplp
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale


# noinspection DuplicatedCode
def prediction(X, w):
    """Get the scaled prediction from the features and weights.

    We should be able to optimize with jit numba.

    In 100 trials using calf_sat.ipynb, we get the
    following mean and std of the difference from
    logistic regression:

    AUC mu, std  [0.0553 0.0171]
    Accuracy mu, std  [0.0859 0.0442]

    Any changes should improve the AUC mu.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        w : weights

    Returns:
        class probabilities

    """
    assert X.shape[1] == len(w), "X or w have the wrong shape"

    feature_range = range(X.shape[1])
    weighted_features = [X[:, i] * w[i] for i in feature_range]
    # Sum the columns and scale to a probability
    # AUC is the same on scaled and unscaled prediction vectors
    y_unscaled = np.sum(weighted_features, 0)
    y_scaled = np.array(minmax_scale(y_unscaled, feature_range=(-1, 1)))
    return y_scaled


# noinspection DuplicatedCode
def sat_weights(X, y, complexity='high', verbose=False):
    """ Get the weights using MILP and SAT.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        complexity : high or medium
        verbose : whether to print state

    Returns:
        weights

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.metrics import roc_auc_score

        # Make a classification problem
        >>> X_d, y_d = make_classification(
        ...    n_samples=100,
        ...    n_features=10,
        ...    n_informative=5,
        ...    n_redundant=3,
        ...    n_classes=2,
        ...    hypercube=True,
        ...    random_state=8
        ... )

        Low complexity throws an exception
        >>> w_d, status_d = sat_weights(X_d, y_d, complexity='low')
        Traceback (most recent call last):
         ...
        ValueError: Complexity must be medium or high to run the SAT solver.

        The status includes variable information that we skip in the doctest:
        Solving with CP-SAT solver v9.6.2534
        Objective value = 510.815934681813
        Problem solved in 33.000000 milliseconds
        The number of constraints with medium complexity is features
        >>> w_d, status_d = sat_weights(X_d, y_d, complexity='medium', verbose=True) # doctest:+ELLIPSIS
        Number of variables = 10
        ...
        Problem solved in 0 iterations
        Problem solved in 0 branch-and-bound nodes
        <BLANKLINE>
        w[0]  =  -1.0
        w[1]  =  -1.0
        w[2]  =  1.0
        w[3]  =  1.0
        w[4]  =  -1.0
        w[5]  =  -1.0
        w[6]  =  -1.0
        w[7]  =  1.0
        w[8]  =  -1.0
        w[9]  =  -1.0

        >>> w_d
        [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0]

        The SAT solver identifies good initial weights.
        >>> auc = roc_auc_score(y_true=y_d, y_score=prediction(X_d, w_d))
        >>> np.round(auc, 2)
        0.89

        High complexity solves with all constraints
        The number of constraints with high complexity is features + samples
        >>> w_d, status_d = sat_weights(X_d, y_d, complexity='high', verbose=True) # doctest:+ELLIPSIS
        Number of variables = 110
        Solving with CP-SAT solver v9.6.2534
        ...
        Problem solved in 0 iterations
        Problem solved in 0 branch-and-bound nodes
        <BLANKLINE>
        w[0]  =  -1.0
        w[1]  =  -1.0
        w[2]  =  1.0
        w[3]  =  0.0
        w[4]  =  -1.0
        w[5]  =  -1.0
        w[6]  =  -1.0
        w[7]  =  -1.0
        w[8]  =  -1.0
        w[9]  =  -1.0

        >>> w_d
        [-1.0, -1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

        The SAT solver with high complexity identifies better initial weights.
        >>> auc = roc_auc_score(y_true=y_d, y_score=prediction(X_d, w_d))
        >>> np.round(auc, 2)
        0.95

    """

    if complexity not in ['medium', 'high']:
        raise ValueError("Complexity must be medium or high to run the SAT solver.")

    feature_range = list(range(X.shape[1]))
    sample_range = list(range(X.shape[0]))

    solver = pywraplp.Solver.CreateSolver('SAT')
    if not solver:
        raise RuntimeError("SAT solver unavailable")

    # weights on a hypercube vertex
    # weights in [-10, 10] give slightly better auc
    # weights in [-100, 100] hang without improved auc.
    w = {}
    for i in feature_range:
        w[i] = solver.IntVar(-1, 1, 'w[%i]' % i)

    # The positive and negative cases
    # "AUC is concerned with ranking, more specifically the
    # probability that a randomly-chosen positive sample is
    # ranked higher than a randomly-chosen negative sample."
    # see https://stats.stackexchange.com/a/423137/346527
    #
    # We can represent the idea that the positive samples
    # will be ranked higher than the negative samples
    # through a sum of the positive samples required to be
    # greater than the sum of the negative samples.
    # sum_pos sums over the true positive and false positive cases
    # while sum_neg sums over the true negative and false negative cases
    pos = sum([X[i][j] * w[j] for j in feature_range for i in sample_range if y[i] == 1])
    neg = sum([X[i][j] * w[j] for j in feature_range for i in sample_range if y[i] == 0])

    # encourage zero
    w_non_zero = sum([1 for i in feature_range if not w[i] == 0])

    # we expect that the sum over the positive cases will be larger than over the negative
    solver.Add(pos >= neg)

    if complexity == 'high':
        # the slack variables, p, significantly increase run-time.
        # if complexity == 'high':
        p = {}
        # sample probability constraints
        for i in sample_range:
            p[i] = solver.NumVar(0, solver.infinity(), 'p[%i]' % i)
            constraint_expr = sum([X[i][j] * w[j] for j in feature_range])
            if y[i] == 1:
                solver.Add(constraint_expr + p[i] >= 0)
            else:
                solver.Add(constraint_expr - p[i] <= 0)
        row_slack = sum([p[i] for i in sample_range])

        # Maximize the difference between the positive and negative cases
        # Minimizing the sum of the coefficient vector is a way to
        # include regularization of the weights, like LASSO.
        # Minimizing the weight sum does not seem to improve AUC.
        # https://scikit-learn.org/stable/modules/linear_model.html#:~:text=The%20lasso%20estimate,the%20coefficient%20vector.
        solver.Maximize(pos - neg - w_non_zero - row_slack)
    else:
        solver.Maximize(pos - neg)

    # solve the classification problem
    status = solver.Solve()

    # save execution status
    result = {
        'num_variables': solver.NumVariables(),
        'solver_version': solver.SolverVersion(),
        'solver_status': pywraplp.Solver.OPTIMAL,
        'objective_value': solver.Objective().Value(),
        'solver_wall_time': solver.wall_time(),
        'solver_iterations': solver.iterations(),
        'solver_nodes': solver.nodes()
    }

    # print execution status as requested
    if verbose:
        print('Number of variables =', solver.NumVariables())
        print(f'Solving with {solver.SolverVersion()}')
        if status == pywraplp.Solver.OPTIMAL:
            print('Objective value =', solver.Objective().Value())
            print('Problem solved in %f milliseconds' % solver.wall_time())
            print('Problem solved in %d iterations' % solver.iterations())
            print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
            print()
            for i in range(len(w)):
                print(w[i].name(), ' = ', w[i].solution_value())
        else:
            print('The problem does not have an optimal solution.')

    weights = [v.solution_value() for v in w.values()]
    return weights, result
