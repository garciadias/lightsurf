from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator

from lightsurf.domain.interfaces.data_reader import DataReaderInterface
from lightsurf.domain.interfaces.training import Controller
from lightsurf.domain.services.data.data_service import DataService
from lightsurf.domain.services.evaluation_service import ModelEvaluator
from lightsurf.domain.services.training_service import (
    FeatureSelector,
    FileModelRepository,
    TrainingService,
)

global RANDOM_STATE
RANDOM_STATE = 31869


@dataclass
class TrainingController(Controller):
    service: TrainingService

    def __post_init__(self):
        self.model = self.service.model_trainer.model

    def fit(self) -> BaseEstimator:
        self.model = self.service.train_model()
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(data)

    def __str__(self) -> str:
        model_name = self.model.__class__.__name__
        model_params = self.model.get_params()
        return f"{model_name}({model_params})"


def create_model_controller(
    data_reader: DataReaderInterface,
    target_variable: str,
    model: BaseEstimator,
    features: list[str] | None = None,
    model_path: str = "data/models/",
    output_filetype: Literal["pkl"] = "pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    average_evaluator: Literal["micro", "macro", "weighted", "binary", None] = None,
    str_threshold: str = "mean",
    n_rows: int | None = None,
    feature_selection: str | None = "select_from_model",
    param_distributions: dict[str, list] | None = None,
) -> TrainingController:
    data_service = DataService(
        data_reader=data_reader,
        target=target_variable,
        features=features,
        test_size=test_size,
        random_state=random_state,
        n_rows=n_rows,
    )
    model_repository = FileModelRepository(
        base_path=model_path, filetype=output_filetype
    )
    model_trainer = model
    if feature_selection:
        feature_selector = FeatureSelector(
            selection_type="select_from_model", params={"threshold": str_threshold}
        )
    else:
        feature_selector = None
    model_evaluator = ModelEvaluator(average=average_evaluator)
    training_service = TrainingService(
        data_service=data_service,
        model_repository=model_repository,
        model_trainer=model_trainer,
        model_evaluator=model_evaluator,
        feature_selector=feature_selector,
        param_distributions=param_distributions,
    )
    controller = TrainingController(training_service)
    return controller
