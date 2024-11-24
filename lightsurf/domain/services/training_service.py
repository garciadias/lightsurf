import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from lightsurf.domain.interfaces.data_reader import DataServiceInterface
from lightsurf.domain.interfaces.training import (
    FeatureSelectionInterface,
    ModelRepositoryInterface,
    TrainerInterface,
    TrainingServiceInterface,
)
from lightsurf.domain.services.evaluation_service import ModelEvaluator


@dataclass
class FeatureSelector(FeatureSelectionInterface):
    selection_type: Literal["sequential", "select_from_model", "rfe"] = "sequential"
    params: Dict[str, Any] = field(default_factory=dict)

    def select_features(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | pd.DataFrame,
        model: BaseEstimator,
    ) -> list[str]:
        self.selector = self._get_selector(X_train, y_train, model)
        self.support_ = self.selector.get_support()
        self.original_features_ = (
            X_train.columns
            if isinstance(X_train, pd.DataFrame)
            else np.arange(X_train.shape[1])
        )
        self.selected_features_ = [
            f for f, s in zip(self.original_features_, self.support_) if s
        ]
        return self.selected_features_

    def _get_selector(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | pd.DataFrame,
        model: BaseEstimator,
    ) -> SequentialFeatureSelector | RFE | SelectFromModel:
        if self.selection_type == "sequential":
            return self._sequential_feature_selector(X_train, y_train, model)
        if self.selection_type == "rfe":
            return self._recursive_feature_elimination(X_train, y_train, model)
        if self.selection_type == "select_from_model":
            return self._select_from_model(X_train, y_train, model)
        raise AttributeError(
            f"Feature selection type {self.selection_type} is not supported"
        )

    def _recursive_feature_elimination(
        self, X_train, y_train, model: BaseEstimator
    ) -> RFE:
        rfe = RFE(estimator=model, **(self.params or {}))
        rfe.fit(X_train, y_train)
        return rfe

    def _select_from_model(
        self, X_train, y_train, model: BaseEstimator
    ) -> SelectFromModel:
        print(model.__class__.__name__)
        sfm = SelectFromModel(estimator=model, **(self.params or {}))
        sfm.fit(X_train, y_train)
        return sfm

    def _sequential_feature_selector(
        self, X_train, y_train, model: BaseEstimator
    ) -> SequentialFeatureSelector:
        sfs_forward = SequentialFeatureSelector(estimator=model, **(self.params or {}))
        sfs_forward.fit(X_train, y_train)
        return sfs_forward


class ModelTrainer:
    def __init__(self, model: BaseEstimator = None, **kwargs):
        self.feature_selector = None
        self.model = model
        if "feature_selector" in kwargs:
            self.feature_selector = kwargs.pop("feature_selector")
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if self.model is not None:
            self.model = self.model.set_params(**kwargs)

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | pd.DataFrame,
    ) -> BaseEstimator:
        logging.info("Training model...")
        logging.info("Model:", self.model)
        logging.info("X_train shape:", X_train.shape)
        logging.info("y_train shape:", y_train.shape)
        if self.feature_selector is not None:
            selected_features = self.feature_selector.select_features(
                X_train, y_train, self.model
            )
            self.model.fit(
                X_train[selected_features],
                y_train,
            )
        else:
            self.model.fit(
                X_train,
                y_train,
            )
        return self.model

    def score(self, X, y):
        return self.model.score(X, y)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            setattr(self.model, key, value)
        return self

    def get_params(self, deep=False):
        if self.model is None:
            return self.__dict__
        return self.model.get_params(deep=deep)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)
        super().__init__(**kwargs)


@dataclass
class TrainingService(TrainingServiceInterface):
    data_service: DataServiceInterface
    model_repository: ModelRepositoryInterface
    model_trainer: TrainerInterface
    model_evaluator: ModelEvaluator
    feature_selector: FeatureSelector | None = None
    identity: str = "model"
    split_by: str | None = None
    sequence_split_by: str | None = None
    param_distributions: Dict[str, list] | None = None

    def train_model(self) -> BaseEstimator:
        # Load data
        self.data_service.load()
        if self.param_distributions is not None:
            hyperparameter_search = RandomizedSearchCV(
                estimator=self.model_trainer,
                param_distributions=self.param_distributions,
                n_iter=1,
                random_state=42,
            )
            hyperparameter_search.fit(
                self.data_service.X_train,
                self.data_service.y_train,
            )
        else:
            hyperparameter_search = None
            self.fit()
        # Evaluate model
        self.model_evaluator.evaluate_model(self.data_service, self.model_trainer)
        # Save model evaluation
        self.model_evaluator.save_evaluation(
            identity=self.identity, base_path=self.model_repository.base_path
        )
        # Save evaluation examples
        # Save model
        self.model_repository.save_model(
            identity=self.identity, model=self.model_trainer.model
        )
        return self.model_trainer.model

    def fit(self, X=None, y=None):
        # Perform feature selection
        if self.feature_selector is not None:
            self.data_service = self.feature_selector.select_features(
                self.data_service, self.model_trainer
            )
        # Train model
        self.model_trainer.fit(self.data_service)

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self.model_trainer.model, key, value)
        return self

    def get_params(self, deep=False):
        return self.model_trainer.model.get_params(deep=deep)

    def predict(self, X):
        return self.model_trainer.model.predict(X)

    def score(self, X, y):
        return self.model_trainer.model.score(X, y)


@dataclass
class FileModelRepository(ModelRepositoryInterface):
    base_path: Union[str, Path]
    filetype: Literal["pkl"] = "pkl"
    model_type: str = field(init=False)

    def __post_init__(self):
        if self.filetype not in ["pkl"]:
            raise AttributeError(f"Filetype {self.filetype} is not supported")

    def save_model(self, identity: str, model: BaseEstimator):
        self.model = model
        path = self._prepare_model_path(identity)
        self.model_type = model.__class__.__name__
        joblib.dump(self, path)

    def load_model(self, identity: str) -> BaseEstimator:
        path = self._prepare_model_path(identity)
        self = joblib.load(path)
        return self.model

    def _prepare_model_path(self, identity: str):
        base_path = Path(self.base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        path = Path(self.base_path).joinpath(f"{identity}.{self.filetype}")
        return path
