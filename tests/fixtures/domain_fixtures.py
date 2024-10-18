import pytest

from lightsurf.domain.services.data.data_service import DataService, FileDataReader
from lightsurf.domain.services.training_service import (
    FeatureSelector,
    FileModelRepository,
    ModelEvaluator,
    ModelTrainer,
    TrainingService,
)


@pytest.fixture
def testing_data_reader(testing_data_path, test_schema):
    return FileDataReader(path=testing_data_path, schema=test_schema)


@pytest.fixture
def testing_model_repository(tmp_path):
    return FileModelRepository(base_path=tmp_path, filetype="pkl")


@pytest.fixture
def testing_feature_selector():
    return FeatureSelector(
        selection_type="sequential", params={"n_features_to_select": 2}
    )


@pytest.fixture
def testing_data_service(testing_data_reader):
    data_service = DataService(
        data_reader=testing_data_reader,
        target="target",
    )
    data_service.load()
    return data_service


@pytest.fixture
def testing_model_trainer(testing_model):
    return ModelTrainer(model=testing_model)


@pytest.fixture
def testing_model_evaluator():
    return ModelEvaluator(average="micro")


@pytest.fixture
def training_service(
    testing_data_service,
    testing_model_repository,
    testing_model_trainer,
    testing_model_evaluator,
):
    return TrainingService(
        data_service=testing_data_service,
        model_repository=testing_model_repository,
        model_trainer=testing_model_trainer,
        model_evaluator=testing_model_evaluator,
    )


@pytest.fixture
def training_service_with_feature_selection(
    testing_data_service,
    testing_model_repository,
    testing_model_trainer,
    testing_model_evaluator,
    testing_feature_selector,
):
    return TrainingService(
        data_service=testing_data_service,
        model_repository=testing_model_repository,
        model_trainer=testing_model_trainer,
        model_evaluator=testing_model_evaluator,
        feature_selector=testing_feature_selector,
    )
