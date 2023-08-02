import pytest
from sklearn.utils.estimator_checks import check_estimator

from sppam.sppam import SPPAM


@pytest.mark.parametrize(
    "estimator",
    [SPPAM()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
