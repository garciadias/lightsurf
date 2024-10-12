from typing import Literal, Optional

from sklearn.base import BaseEstimator

from lightsurf.domain.controllers.controller import TrainingController
from lightsurf.domain.interfaces.data_reader import DataReaderInterface
from lightsurf.domain.services.data.data_service import (
    DataService,
)
from lightsurf.domain.services.training_service import (
    FeatureSelector,
    FileModelRepository,
    ModelEvaluator,
    ModelTrainer,
    TrainingService,
)


def create_model_controller(
    data_reader: DataReaderInterface,
    target_variable: str,
    model: BaseEstimator,
    model_path: str = "data/models/",
    output_filetype: Literal["pkl"] = "pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    average_evaluator: Literal["micro", "macro", "weighted", "binary", None] = None,
    str_threshold: str = "mean",
    n_rows: Optional[int] = None,
    feature_selection: Optional[str] = "select_from_model",
    split_by: Optional[str] = None,
    sequence_split_by: Optional[str] = None,
):
    data_service = DataService(
        data_reader=data_reader,
        target=target_variable,
        test_size=test_size,
        random_state=random_state,
        n_rows=n_rows,
    )
    model_repository = FileModelRepository(
        base_path=model_path, filetype=output_filetype
    )
    model_trainer = ModelTrainer(model=model)
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
        split_by=split_by,
        sequence_split_by=sequence_split_by,
    )
    controller = TrainingController(training_service)
    return controller
