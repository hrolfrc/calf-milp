import pytest
from sklearn.utils.estimator_checks import check_estimator

from calf_milp import CalfMilp


@pytest.mark.parametrize(
    "estimator",
    [CalfMilp()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
