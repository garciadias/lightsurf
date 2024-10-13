import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from lightsurf.domain.interfaces.data_models.evaluation import (
    ModelEvaluationClassification,
)
from lightsurf.domain.interfaces.data_reader import DataServiceInterface
from lightsurf.domain.interfaces.training import (
    ModelRepositoryInterface,
    TrainingServiceInterface,
)


@dataclass
class ModelTrainer:
    model: BaseEstimator

    def train_model(self, data_service: DataServiceInterface) -> BaseEstimator:
        print("Training model...")
        self.model.fit(
            data_service.X_train,
            data_service.y_train,
        )
        return self.model


@dataclass
class ModelEvaluator:
    average: Literal["micro", "macro", "weighted", "binary", None] = None

    def evaluate_model(
        self, data_service: DataServiceInterface, model_trainer: ModelTrainer
    ):
        self.y_pred = model_trainer.model.predict(data_service.X_test).astype(bool)
        self.y_pred = pd.Series(
            self.y_pred, index=data_service.test_index[: len(self.y_pred)]
        ).astype(bool)
        self.y_pred_train = model_trainer.model.predict(data_service.X_train).astype(
            bool
        )
        self.y_pred_train = pd.Series(
            self.y_pred_train, index=data_service.train_index[: len(self.y_pred_train)]
        ).astype(bool)
        accuracy = accuracy_score(data_service.y_test, self.y_pred)
        precision = precision_score(
            data_service.y_test, self.y_pred, average=self.average
        )
        recall = recall_score(data_service.y_test, self.y_pred, average=self.average)
        f1 = f1_score(data_service.y_test, self.y_pred, average=self.average)
        self.model_evaluation = ModelEvaluationClassification(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )
        self.confusion_matrix = self.create_confusion_matrix(
            data_service.y_test, self.y_pred
        )
        self.indexes = self.get_train_test_indexes(data_service)
        return self.model_evaluation

    def save_evaluation(
        self, identity: Union[str, Path], base_path: Union[str, Path]
    ) -> str:
        base_path = Path(base_path)
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        evaluation_path = f"{base_path}/{identity}_evaluation.json"
        with open(evaluation_path, "w") as f:
            json.dump(self.model_evaluation.model_dump(), f)
        self.confusion_matrix.savefig(
            f"{base_path}/{identity}_confusion_matrix.png", dpi=300
        )
        plt.close(self.confusion_matrix)
        self.indexes.to_csv(f"{base_path}/{identity}_train_test_split.csv", index=False)
        return evaluation_path

    def get_train_test_indexes(
        self, data_service: DataServiceInterface
    ) -> pd.DataFrame:
        train_pred = pd.DataFrame(self.y_pred_train, columns=["y_pred"])
        train_pred["y_true"] = data_service.y_train
        train_pred["set"] = "train"
        test_pred = pd.DataFrame(self.y_pred, columns=["y_pred"])
        test_pred["y_true"] = data_service.y_test
        test_pred["set"] = "test"
        indexes = pd.concat([train_pred, test_pred])
        indexes.index.name = "index"
        return indexes.reset_index()

    def save_dataset_examples(
        self,
        identity: Union[str, Path],
        base_path: Union[str, Path],
        data_service: DataServiceInterface,
        n_examples_per_outcome: int = 2,
    ):
        base_path = Path(base_path)
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        dataset_path = f"{base_path}/{identity}_dataset_sample.csv"
        correct_predictions = (data_service.y_test == self.y_pred).values
        wrong_predictions = ~correct_predictions
        sample_columns = ["y_test", "y_pred"]
        sample_columns += list(data_service.X_test.columns[:3])
        dataset = pd.concat(
            [
                data_service.y_test,
                self.y_pred,
                data_service.X_test[data_service.X_test.columns[:3]],
            ],
            axis=1,
        )
        dataset.index = data_service.X_test.index
        dataset.columns = pd.Index(sample_columns)
        sample_data = pd.concat(
            [
                dataset[correct_predictions & (dataset.y_test == outcome)]
                .head(n_examples_per_outcome)
                .sort_values("y_test")
                for outcome in dataset["y_test"].unique()
            ]
        )
        sample_data = pd.concat(
            [sample_data]
            + [
                dataset[wrong_predictions & (dataset.y_test == outcome)]
                .head(n_examples_per_outcome)
                .sort_values("y_test")
                for outcome in dataset["y_test"].unique()
            ]
        )
        sample_data.reset_index(inplace=True)
        sample_data.to_csv(dataset_path, index=False)
        return dataset_path

    def create_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        try:
            cm = pd.DataFrame(cm, index=range(1, 6), columns=range(1, 6))
        except ValueError:
            print("Unique values in y_test:", np.unique(y_test))
            print("Unique values in y_pred:", np.unique(y_test))
        fig, ax = plt.subplots(1, 1, figsize=(9 * 0.7, 9 * 0.7))
        sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        return fig


@dataclass
class FeatureSelector:
    selection_type: Literal["sequential", "select_from_model", "rfe"] = "sequential"
    params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def select_features(
        self,
        data_service: DataServiceInterface,
        model_trainer: ModelTrainer,
    ) -> DataServiceInterface:
        self.selector = self._get_selector(data_service, model_trainer)
        self.support_ = self.selector.get_support()
        self.original_features_ = data_service.X_train.columns
        self.selected_features_ = [
            f for f, s in zip(self.original_features_, self.support_) if s
        ]
        data_service = self._modify_data_service(data_service)
        return data_service

    def _modify_data_service(self, data_service):
        data_service.X_train = data_service.X_train[self.selected_features_]
        data_service.X_test = data_service.X_test[self.selected_features_]
        return data_service

    def _get_selector(
        self, data_service: DataServiceInterface, model_trainer: ModelTrainer
    ):
        if self.selection_type == "sequential":
            return self._sequential_feature_selector(data_service, model_trainer)
        if self.selection_type == "rfe":
            return self._recursive_feature_elimination(data_service, model_trainer)
        if self.selection_type == "select_from_model":
            return self._select_from_model(data_service, model_trainer)
        raise AttributeError(
            f"Feature selection type {self.selection_type} is not supported"
        )

    def _recursive_feature_elimination(
        self, data_service: DataServiceInterface, model_trainer: ModelTrainer
    ) -> RFE:
        rfe = RFE(estimator=model_trainer.model, **(self.params or {}))
        rfe.fit(data_service.X_train, data_service.y_train)
        return rfe

    def _select_from_model(
        self, data_service: DataServiceInterface, model_trainer: ModelTrainer
    ) -> SelectFromModel:
        sfm = SelectFromModel(estimator=model_trainer.model, **(self.params or {}))
        sfm.fit(data_service.X_train, data_service.y_train)
        return sfm

    def _sequential_feature_selector(
        self, data_service: DataServiceInterface, model_trainer: ModelTrainer
    ) -> SequentialFeatureSelector:
        sfs_forward = SequentialFeatureSelector(
            estimator=model_trainer.model, **(self.params or {})
        )
        sfs_forward.fit(data_service.X_train, data_service.y_train)
        return sfs_forward


@dataclass
class TrainingService(TrainingServiceInterface):
    data_service: DataServiceInterface
    model_repository: ModelRepositoryInterface
    model_trainer: ModelTrainer
    model_evaluator: ModelEvaluator
    feature_selector: Optional[FeatureSelector] = None
    identity: str = "model"
    split_by: Optional[str] = None
    sequence_split_by: Optional[str] = None

    def train_model(self) -> BaseEstimator:
        # Load data
        self.data_service.load()
        # Perform feature selection
        if self.feature_selector is not None:
            self.data_service = self.feature_selector.select_features(
                self.data_service, self.model_trainer
            )
        # Train model
        self.model_trainer.train_model(self.data_service)
        # Evaluate model
        self.model_evaluator.evaluate_model(self.data_service, self.model_trainer)
        # Save model evaluation
        self.model_evaluator.save_evaluation(
            identity=self.identity, base_path=self.model_repository.base_path
        )
        # Save evaluation examples
        # self.model_evaluator.save_dataset_examples(
        #     identity=self.identity,
        #     base_path=self.model_repository.base_path,
        #     data_service=self.data_service,
        # )
        # Save model
        self.model_repository.save_model(
            identity=self.identity, model=self.model_trainer.model
        )
        return self.model_trainer.model


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
