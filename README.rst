.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sppam/badge/?version=latest
.. _ReadTheDocs: https://sppam.readthedocs.io/en/latest/?badge=latest

SPPAM - Saddle point problem for AUC maximization
============================================================

An AUC optimizing binomial classifier.

Contact
------------------
Rolf Carlson hrolfrc@gmail.com

Install
------------------
Use pip to install sppam.

``pip install sppam``

Introduction
------------------
This is a python implementation of a classifier that endeavors to solve the `saddle point problem for AUC maximization`_. [1]

SPPAM provides classification and prediction for two classes, the binomial case.  Small to medium problems are supported.  This is research code and a work in progress.

SPPAM is designed for use with scikit-learn_ pipelines and composite estimators.

.. _scikit-learn: https://scikit-learn.org

.. _`saddle point problem for AUC maximization`: https://www.frontiersin.org/articles/10.3389/fams.2019.00030/full

Example
------------------

.. code:: ipython2

    from sppam import SPPAM
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

Make a classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    cls = SPPAM().fit(X_train, y_train)

Get the score on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)


.. parsed-literal::

    1.0

References
------------------
[1] Natole Jr, Michael & Ying, Yiming & Lyu, Siwei. (2019).
Stochastic AUC Optimization Algorithms With Linear Convergence.
Frontiers in Applied Mathematics and Statistics. 5. 10.3389/fams.2019.00030.
