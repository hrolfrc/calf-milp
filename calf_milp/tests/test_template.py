""" Tests for CalfMilp

===============================================================
Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
License: 3-clause BSD
===============================================================


"""
import pytest
from sklearn.datasets import load_iris

from calf_milp import CalfMilp


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_CalfMilp(data):
    X, y = data
    clf = CalfMilp()

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
