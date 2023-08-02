# Quick Start

```python
from sppam import SPPAM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

#### Make a classification problem

```python
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
```

#### Train the classifier

```python
cls = SPPAM().fit(X_train, y_train)
```

#### Get the score on unseen data

```python
cls.score(X_test, y_test)
```

    1.0


