"""
Mixed integer-linear program for classification

inspired by calf where the weights are restricted to vertices on a hypercube.

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .calf_sat import sat_weights


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
    """ The CalfMilp (Saddle Point Problem for AUC Maximization) classifier

        Attributes
        ----------

        classes_ : ndarray of shape (n_classes, )
            A list of class labels known to the classifier.

        coef_ : feature weights of shape (1, n_features).

        n_features_in_ : int
            The number of features of the data passed to :meth:`fit`.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """ Fit the model according to the given training data.

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
        self.w_, self.status_ = sat_weights(X, y)
        self.is_fitted_ = True
        self.coef_ = self.w_
        return self

    def decision_function(self, X):
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

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }
