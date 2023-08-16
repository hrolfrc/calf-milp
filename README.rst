.. -*- mode: rst -*-

|CircleCI|_ |ReadTheDocs|_

.. |CircleCI| image:: https://circleci.com/gh/hrolfrc/calf-milp.svg?style=shield
.. _CircleCI: https://circleci.com/gh/hrolfrc/calf-milp

.. |ReadTheDocs| image:: https://readthedocs.org/projects/calf-milp/badge/?version=latest
.. _ReadTheDocs: https://calf-milp.readthedocs.io/en/latest/?badge=latest

CalfMilp
#####################################

CalfMilp is a binomial classifier that implements a course approximation linear function by mixed integer linear programming.

Contact
------------------
Rolf Carlson hrolfrc@gmail.com

Install
------------------
Use pip to install calf-milp.

``pip install calf-milp``

Introduction
------------------
CalfMilp provides classification and prediction for two classes, the binomial case.  Small problems are supported.  This is research code and a work in progress.

CalfMilp is designed for use with scikit-learn_ pipelines and composite estimators.

.. _scikit-learn: https://scikit-learn.org

Example
===========

.. code:: ipython2

    from calf_milp import CalfMilp
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

Make a classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    seed = 42
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

Train the classifier
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls = CalfMilp().fit(X_train, y_train)

Get the score on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)




.. parsed-literal::

    0.875


