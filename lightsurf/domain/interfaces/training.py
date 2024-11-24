from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, Literal, Union

from sklearn.base import BaseEstimator

from lightsurf.domain.interfaces.data_reader import DataServiceInterface


class FeatureSelectionInterface(ABC):
    selection_type: Literal["sequential", "select_from_model", "rfe"] = "sequential"
    params: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def select_features(self, X_train, y_train, model: BaseEstimator) -> list[str]: ...


class TrainerInterface(ABC):
    model: BaseEstimator
    feature_selector: FeatureSelectionInterface | None

    @abstractmethod
    def __init__(self, model: BaseEstimator): ...

    @abstractmethod
    def fit(self, X_train, y_train) -> BaseEstimator: ...


class TrainingServiceInterface(ABC):
    @abstractmethod
    def __init__(self, data_service: DataServiceInterface): ...

    @abstractmethod
    def train_model(self) -> BaseEstimator: ...

    def evaluate_model(
        self, model_trainer: TrainerInterface, data_service: DataServiceInterface
    ) -> None: ...


class Controller(ABC):
    @abstractmethod
    def __init__(self, training_service: TrainingServiceInterface): ...

    @abstractmethod
    def fit(self) -> BaseEstimator: ...


class ModelRepository(ABC):
    base_path: str

    @abstractmethod
    def save_model(self, identity: str, model: BaseEstimator) -> None: ...

    @abstractmethod
    def load_model(self, identity: str) -> BaseEstimator: ...


class ModelRepositoryInterface(ABC):
    base_path: Union[str, Path]

    @abstractmethod
    def save_model(self, identity: str, model: BaseEstimator) -> None: ...

    @abstractmethod
    def load_model(self, identity: str) -> BaseEstimator: ...
