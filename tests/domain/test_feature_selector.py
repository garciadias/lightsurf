import pytest
from pandas.core.frame import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from lightsurf.domain.services.training_service import FeatureSelector, ModelTrainer


@pytest.fixture
def model_trainer():
    return ModelTrainer(DummyClassifier())


def test_feature_selector_construct():
    selector = FeatureSelector()
    assert selector is not None


def test_data_fixtures(X_train, y_train):
    assert len(X_train) == len(y_train)
    assert len(X_train) > 0
    assert isinstance(X_train, DataFrame)


@pytest.mark.parametrize(
    ("selection_type", "model"),
    [
        ("sequential", DummyClassifier()),
        (
            "sequential",
            RandomizedSearchCV(
                XGBClassifier(),
                param_distributions={},
                n_iter=1,
                cv=2,
                n_jobs=-1,
            ),
        ),
        ("select_from_model", XGBClassifier()),
        ("rfe", XGBClassifier()),
        ("bad_selection", DummyClassifier()),
    ],
)
def test_feature_selector_select_features(testing_data_service, selection_type, model):
    selector = FeatureSelector(selection_type=selection_type)

    if selection_type == "bad_selection":
        with pytest.raises(AttributeError):
            selector.select_features(
                testing_data_service.X_train, testing_data_service.y_train, model
            )
        return  # Exit early
    X_train, y_train = testing_data_service.X_train, testing_data_service.y_train
    features = selector.select_features(X_train, y_train, model)
    assert len(features) > 0
    assert len(features) < len(X_train.columns)
