from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from sklearn.base import BaseEstimator

from lightsurf.domain.interfaces.data_reader import DataServiceInterface


class TrainingServiceInterface(ABC):
    @abstractmethod
    def __init__(self, data_service: DataServiceInterface): ...

    @abstractmethod
    def train_model(self) -> BaseEstimator: ...


class Controller(ABC):
    @abstractmethod
    def __init__(self, training_service: TrainingServiceInterface): ...

    @abstractmethod
    def train(self) -> BaseEstimator: ...


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
