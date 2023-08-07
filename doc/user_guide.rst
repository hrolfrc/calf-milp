.. title:: User guide : contents

.. _user_guide:
User Guide
==========

Make a classification problem
-----------------------------

.. code:: ipython2

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from calf_milp import CalfMilp

.. code:: ipython2

    seed = 45
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
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    cls = CalfMilp().fit(X_train, y_train)

Get the score for class prediction on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)




.. parsed-literal::

    0.76



Class probabilities
^^^^^^^^^^^^^^^^^^^

We vertically stack the ground truth on the top with the probabilities
of class 1 on the bottom. The first five entries are shown.

.. code:: ipython2

    np.round(np.vstack((y_train, cls.predict_proba(X_train).T))[:, 0:5], 2)




.. parsed-literal::

    array([[1.  , 1.  , 0.  , 0.  , 0.  ],
           [0.23, 0.3 , 0.86, 0.47, 0.67],
           [0.77, 0.7 , 0.14, 0.53, 0.33]])



.. code:: ipython2

    roc_auc_score(y_true=y_train, y_score=cls.predict_proba(X_train)[:, 1])




.. parsed-literal::

    0.9630156472261735



Predict the classes
^^^^^^^^^^^^^^^^^^^

The ground truth is on the top and the predicted classes are on the
bottom. The first five entries are shown.

.. code:: ipython2

    y_pred = cls.predict(X_test)
    np.vstack((y_test, y_pred))[:, 0:5]




.. parsed-literal::

    array([[0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1]])



The class prediction is expected to be lower than the auc prediction.

.. code:: ipython2

    roc_auc_score(y_true=y_test, y_score=y_pred)




.. parsed-literal::

    0.7532051282051281


