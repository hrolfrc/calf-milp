#####################################
Quick Start
#####################################

SPPAM performs binomial classification.

Quick Start
===========

.. code:: ipython2

    from sppam import SPPAM
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import numpy as np

.. code:: ipython2

    seed = 43
    X, y = make_classification(
        n_samples=100,
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

    0.88
