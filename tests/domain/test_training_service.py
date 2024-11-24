import pytest
from sklearn.dummy import DummyClassifier

from lightsurf.domain.services import training_service as ts


def test_file_model_repository_filetype(tmp_path):
    with pytest.raises(AttributeError):
        ts.FileModelRepository(base_path=tmp_path, filetype="wrong")


def test_model_trainer(testing_data_service):
    trainer = ts.ModelTrainer(DummyClassifier())
    X_train, y_train = testing_data_service.X_train, testing_data_service.y_train
    trainer.fit(X_train, y_train)
    assert trainer.model is not None
    assert trainer.model.predict([[0, 0]]) is not None
