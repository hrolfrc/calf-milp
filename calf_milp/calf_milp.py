"""

The CalfMilp classifier.

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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from calf_sat import sat_weights


def scaled_predict(X, w):
    return np.array(
        minmax_scale(
            predict(X, w),
            feature_range=(-1, 1)
        )
    )


def predict(X, w):
    return np.sum(np.multiply(X, w), 1)


# noinspection PyAttributeOutsideInit
class CalfMilp(ClassifierMixin, BaseEstimator):
    """ The CalfMilp classifier

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
            >>> from calfcv import Calf
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
        self.is_fitted_ = True
        self.coef_ = self.w_
        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the decision vector (n_samples)

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

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix for which we want to get the predictions.

        Returns:
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

        Parameters:

            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where n_samples is the number of samples and
                n_features is the number of features.

        Returns:

            T: array-like of shape (n_samples, n_classes)
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

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        return X[:, np.asarray(self.coef_).nonzero()]

    def fit_transform(self, X, y):
        """ Fit to the data, then reduce X to the features that contribute positive AUC.

            Arguments:
                X : array-like, shape (n_samples, n_features)
                    The training input features and samples

                y : array-like of shape (n_samples,)

            Returns:
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
