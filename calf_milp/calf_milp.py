"""
The CalfMilp classifier.

===============================================================
Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
License: 3-clause BSD
===============================================================


Mixed integer-linear program for classification.  CalfMilp is
based on the idea from Calf that the weights are restricted to
vertices on a hypercube, or zero. [1]

References
========================
[1] Jeffries, C.D., Ford, J.R., Tilson, J.L. et al.
A greedy regression algorithm with coarse weights offers novel advantages.
Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2

"""
import time

import numpy as np
from ortools.linear_solver import pywraplp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections import Counter


def predict(X, w):
    return np.sum(np.multiply(X, w), 1)


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
        >>> auc = roc_auc_score(y_true=y_d, y_score=predict(X_d, w_d))
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
        w[2]  =  0.0
        w[3]  =  0.0
        w[4]  =  0.0
        w[5]  =  1.0
        w[6]  =  0.0
        w[7]  =  -1.0
        w[8]  =  -1.0
        w[9]  =  -1.0

        >>> w_d
        [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, -1.0]

        The SAT solver with high complexity identifies better initial weights.
        >>> auc = roc_auc_score(y_true=y_d, y_score=predict(X_d, w_d))
        >>> np.round(auc, 2)
        0.99

    """

    if complexity not in ['medium', 'high']:
        raise ValueError("Complexity must be medium or high to run the SAT solver.")

    feature_range = list(range(X.shape[1]))
    sample_range = list(range(X.shape[0]))

    solver = pywraplp.Solver.CreateSolver('SAT')
    if not solver:
        raise RuntimeError("SAT solver unavailable")
    solver.SetTimeLimit(1000)

    # n_plus and n_minus are the numbers of samples of the positive and negative cases.
    # protect against division by zero.
    n_plus = max(Counter(y)[1], 1)
    n_minus = max(Counter(y)[0], 1)

    w = {}
    for i in feature_range:
        w[i] = solver.IntVar(-1, 1, 'w[%i]' % i)

    pos = (1 / n_plus) * sum([X[i][j] * w[j] for j in feature_range for i in sample_range if y[i] == 1])
    neg = (1 / n_minus) * sum([X[i][j] * w[j] for j in feature_range for i in sample_range if y[i] == 0])

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
                solver.Add(constraint_expr + p[i] >= 1)
            else:
                solver.Add(constraint_expr - p[i] <= -1)
        row_slack = sum([p[i] for i in sample_range])

        solver.Maximize(pos - neg - row_slack)
    else:
        solver.Maximize(pos - neg)

    # solve the classification problem
    try:
        status = solver.Solve()
    except Exception:
        result = {}
        weights = [0] * len(w)
        pass
    else:
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


# noinspection PyAttributeOutsideInit
class CalfMilp(ClassifierMixin, BaseEstimator):
    """ The CalfMilp classifier

        Attributes
        ----------

        coef_ : array of shape (n_features, )
            Estimated coefficients for the linear fit problem.  Only
            one target should be passed, and this is a 1D array of length
            n_features.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        classes_ : list
            The unique class labels

        fit_time_ : float
            The number of seconds to fit X to y

        Notes
        -----

        The feature matrix must be centered at 0.  This can be accomplished with
        sklearn.preprocessing.StandardScaler, or similar.  No intercept is calculated.

        Examples
        --------

            >>> import numpy
            >>> from sklearn.datasets import make_classification as mc
            >>> X, y = mc(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
            >>> numpy.round(X[0:3, :], 2)
            array([[ 1.23, -0.76],
                   [ 0.7 , -1.38],
                   [ 2.55,  2.5 ]])

            >>> y[0:3]
            array([0, 0, 1])

            >>> cls = CalfMilp().fit(X, y)
            >>> cls.score(X, y)
            0.87

            >>> cls.coef_
            [0.0, 1.0]

            >>> numpy.round(cls.score(X, y), 2)
            0.87

            >>> cls.fit_time_ > 0
            True

            >>> cls.predict(np.array([[3, 5]]))
            array([0])

            >>> cls.predict_proba(np.array([[3, 5]]))
            array([[1., 0.]])

        """

    def __init__(self):
        pass

    def fit(self, X, y):
        """ Fit CalfMilp to the training data.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            self
                Fitted estimator.

        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # fit and time the fit
        start = time.time()
        self.w_, self.status_ = sat_weights(X, y)
        self.fit_time_ = time.time() - start
        self.coef_ = self.w_
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            y_d : the decision vector (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = np.array(
            minmax_scale(
                predict(X, self.w_),
                feature_range=(-1, 1)
            )
        )
        return scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix for which we want to get the predictions.

        Returns
        -------
            y_pred : ndarray of shape (n_samples,)
                Vector containing the class labels for each sample.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        if len(self.classes_) < 2:
            y_class = self.y_
        else:
            # and convert to [0, 1] classes.
            y_class = np.heaviside(self.decision_function(X), 0).astype(int)
            # get the class labels
            y_class = [self.classes_[x] for x in y_class]
        return np.array(y_class)

    def predict_proba(self, X):
        """Probability estimates for samples in X.

        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where n_samples is the number of samples and
                n_features is the number of features.

        Returns
        -------
            T : array-like of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in the model,
                where classes are ordered as they are in `self.classes_`.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        y_proba = np.array(
            minmax_scale(
                self.decision_function(X),
                feature_range=(0, 1)
            )
        )
        class_prob = np.column_stack((1 - y_proba, y_proba))
        return class_prob

    def transform(self, X):
        """ Reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        return X[:, np.asarray(self.coef_).nonzero()]

    def fit_transform(self, X, y):
        """ Fit to the data, then reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }
