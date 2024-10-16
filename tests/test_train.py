import pytest
from sklearn.model_selection import RandomizedSearchCV

from lightsurf.domain.controllers.deep_controller import train_lstm_regressor
from lightsurf.domain.controllers.shallow_controller import train_xgboost_regressor


@pytest.mark.slow
def test_xgboost_classifier():
    controller = train_xgboost_regressor(n_rows=100)
    assert controller is not None


@pytest.mark.slow
def test_lstm():
    controller = train_lstm_regressor(n_rows=100)
    assert controller is not None
    model_repository = controller.service.model_repository
    model = model_repository.load_model("model")
    assert model is not None
    assert isinstance(model, RandomizedSearchCV)
    X_test = controller.service.data_service.X_test
    y_test = controller.service.data_service.y_test
    y_pred = controller.predict(X_test)
    model.score(X_test, y_test)
    assert y_pred is not None
