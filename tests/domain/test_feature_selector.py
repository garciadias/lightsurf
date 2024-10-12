import pytest
from pandas.core.frame import DataFrame
from sklearn.dummy import DummyClassifier

from lightsurf.domain.services.training_service import FeatureSelector, ModelTrainer


@pytest.fixture
def model_trainer():
    return ModelTrainer(DummyClassifier())


def test_feature_selector():
    selector = FeatureSelector()
    assert selector is not None


def test_data_fixtures(X_train, y_train):
    assert len(X_train) == len(y_train)
    assert len(X_train) > 0
    assert isinstance(X_train, DataFrame)
