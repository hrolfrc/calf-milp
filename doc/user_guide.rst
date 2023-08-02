.. title:: User guide : contents

.. _user_guide:

User Guide
==========

Make a classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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



Class probabilities
^^^^^^^^^^^^^^^^^^^

We vertically stack the ground truth on the top with the probabilities
of class 1 on the bottom. The first five entries are shown.

.. code:: ipython2

    np.round(np.vstack((y_train, cls.predict_proba(X_train).T))[:, 0:5], 2)




.. parsed-literal::

    array([[0.  , 1.  , 1.  , 0.  , 0.  ],
           [0.65, 0.39, 0.3 , 0.96, 0.83],
           [0.35, 0.61, 0.7 , 0.04, 0.17]])



.. code:: ipython2

    roc_auc_score(y_true=y_train, y_score=cls.predict_proba(X_train)[:, 1])




.. parsed-literal::

    0.9544159544159544



Predict the classes
^^^^^^^^^^^^^^^^^^^

The ground truth is on the top and the predicted class is on the bottom.
The first column is the index. The first five entries are shown.

.. code:: ipython2

    y_pred = cls.predict(X_test)
    np.vstack((y_test, y_pred))[:, 0:5]




.. parsed-literal::

    array([[1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1]])



The class prediction is expected to be lower than the auc prediction.

.. code:: ipython2

    roc_auc_score(y_true=y_test, y_score=y_pred)




.. parsed-literal::

    0.8733766233766235


