"""

======================================
Compare SPPAM with LogisticRegression
======================================

A comparison of LogisticRegression and :class:`SPPAM`
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sppam import SPPAM

logit_auc = []
logit_acc = []
sppam_auc = []
sppam_acc = []

rng = np.random.RandomState(11)
for _ in range(20):
    # Make a classification problem
    X, y_d = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        hypercube=True,
        random_state=rng
    )
    scaler = StandardScaler()
    X_d = scaler.fit_transform(X)

    for desc, clf in [('logit', LogisticRegression(max_iter=10000)), ('SPPAM', SPPAM())]:
        lp = clf.fit(X_d, y_d).predict_proba(X_d)
        auc = roc_auc_score(y_true=y_d, y_score=clf.fit(X_d, y_d).predict_proba(X_d)[:, 1])
        acc = accuracy_score(y_true=y_d, y_pred=clf.fit(X_d, y_d).predict(X_d))
        print(desc, np.round((auc, acc), 2))
        if desc == 'logit':
            logit_auc.append(auc)
            logit_acc.append(acc)
        else:
            sppam_auc.append(auc)
            sppam_acc.append(acc)

# compare the mean of the differences of auc
diff = np.subtract(logit_auc, sppam_auc)

# plot the results
fig, axs = plt.subplots(3, 1, layout='constrained')
xdata = np.arange(len(logit_acc))  # make an ordinal for this
axs[0].plot(xdata, logit_auc, label='LogisticRegression')
axs[0].plot(xdata, sppam_auc, label='SPPAM')
axs[0].set_title('Comparison of SPPAM and LogisticRegression')
axs[0].set_ylabel('AUC')
axs[0].legend()

axs[1].plot(xdata, logit_acc, label='LogisticRegression')
axs[1].plot(xdata, sppam_acc, label='SPPAM')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

axs[2].hist(diff)
axs[2].set_ylabel('AUC difference')
stats = pd.DataFrame(diff).describe().loc[['mean', 'std']].to_string(header=False)
axs[2].text(.1, 2, stats)
fig.set_size_inches(18.5, 20)
plt.show()
