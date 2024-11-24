import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from lightsurf.constants import COLORS
from lightsurf.domain.interfaces.data_models.evaluation import (
    ModelEvaluationClassification,
    ModelEvaluationRegression,
)
from lightsurf.domain.interfaces.data_reader import DataServiceInterface
from lightsurf.domain.interfaces.training import TrainerInterface


def bland_altman_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """Create a Bland–Altman plot to compare the agreement between the predicted and the
    real data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the predicted and real values to compare.
    target : str, optional
        The target variable to compare, by default "normalised_sales"
    dataset_var : str, optional
        The variable containing the dataset information, by default "dataset". This
        variable is used to color the plot and to calculate the mean difference and the
        standard deviation. The training dataset is not used to calculate the mean
        difference and the standard deviation and is shown in a different color to the
        validation dataset.
        Systematic differences between the behavior of the training and validation
        datasets can be identified.

    Returns
    -------
    plt.Figure
        The Bland–Altman plot.
    """
    # Bland–Altman plot
    df = pd.DataFrame.from_dict({
        "predicted": y_pred,
        "test": y_true,
    })
    df.loc[:, "pred_test_mean"] = (df["predicted"] + df["test"]) / 2
    df.loc[:, "pred_test_diff"] = df["predicted"] - df["test"]

    fig, ax = plt.subplots(1, 1, figsize=(16 * 0.7, 9 * 0.7))
    sns.scatterplot(
        data=df,
        x="pred_test_mean",
        y="pred_test_diff",
        ax=ax,
    )
    mean_diff = df["pred_test_diff"].mean()
    std_diff = df["pred_test_diff"].std()
    ax.axhline(0, color=COLORS["green"], linestyle="-")
    ax.axhline(
        mean_diff,
        color=COLORS["red"],
        linestyle="--",
        label="$\\mu_{X_{Pred} - X_{True}}$" + f"{mean_diff:.3f}",
    )
    ax.axhline(
        mean_diff + 1.96 * std_diff,
        color=COLORS["yellow"],
        linestyle="--",
        label=f"$\\pm 1.96 \\sigma = ${1.96 * std_diff:.3f}",
    )
    ax.axhline(
        mean_diff - 1.96 * std_diff,
        color=COLORS["yellow"],
        linestyle="--",
    )
    ax.set_xlabel("$\\frac{X_{True} + X_{Pred}}{2}$", fontsize=20)
    ax.set_ylabel("$X_{Pred} - X_{True}$", fontsize=20)
    ax.set_title("Bland–Altman plot", fontsize=20)
    xlim = plt.xlim()
    ylim = plt.ylim()
    ax.set_ylim(-1 * max(np.abs(ylim)), max(np.abs(ylim)))
    xlim = plt.xlim()
    ylim = plt.ylim()
    mae = np.abs(df["pred_test_diff"]).mean()
    mse = (df["pred_test_diff"] ** 2).mean()
    plt.text(
        xlim[0] + (xlim[1] - xlim[0]) * 0.05,
        ylim[1] - (ylim[1] - ylim[0]) * 0.1,
        f"$\\sigma =${std_diff:.3f}\n$MAE =${mae:.3f}\n$MSE =${mse:.3f}",
        fontsize=16,
        color="black",
    )
    plt.legend()
    plt.tight_layout()
    return fig


@dataclass
class ModelEvaluator:
    average: Literal["micro", "macro", "weighted", "binary", None] = None
    model_type: Literal["classification", "regression"] = "regression"
    model_evaluation: ModelEvaluationClassification | ModelEvaluationRegression = field(
        init=False
    )

    def evaluate_model(
        self, data_service: DataServiceInterface, model_trainer: TrainerInterface
    ):
        if self.model_type == "classification":
            self.evaluate_classification_model(data_service, model_trainer)
        elif self.model_type == "regression":
            self.evaluate_regression_model(data_service, model_trainer)
        else:
            raise AttributeError(f"Model type {self.model_type} is not supported")

    def evaluate_classification_model(
        self, data_service: DataServiceInterface, model_trainer: TrainerInterface
    ):
        self.y_pred = model_trainer.model.predict(data_service.X_test)
        self.y_pred = pd.Series(
            self.y_pred, index=data_service.test_index[: len(self.y_pred)]
        )
        self.y_pred_train = model_trainer.model.predict(data_service.X_train)
        self.y_pred_train = pd.Series(
            self.y_pred_train, index=data_service.train_index[: len(self.y_pred_train)]
        )
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

    def evaluate_regression_model(
        self, data_service: DataServiceInterface, model_trainer: TrainerInterface
    ):
        self.y_pred = model_trainer.model.predict(data_service.X_test)
        self.y_pred_train = model_trainer.model.predict(data_service.X_train)
        mae = mean_absolute_error(data_service.y_test, self.y_pred)
        rmse = root_mean_squared_error(data_service.y_test, self.y_pred)
        r2 = r2_score(data_service.y_test, self.y_pred)
        self.model_evaluation = ModelEvaluationRegression(
            mean_absolute_error=mae,
            mean_squared_error=rmse,
            r2_score=r2,
        )
        self.indexes = self.get_train_test_indexes(data_service)
        print("Creating bland altman plot")
        self.bland_altman_plot = bland_altman_plot(data_service.y_test, self.y_pred)
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
        if self.model_type == "classification":
            self.confusion_matrix.savefig(
                f"{base_path}/{identity}_confusion_matrix.png", dpi=300
            )
            plt.close(self.confusion_matrix)
        if self.model_type == "regression":
            self.bland_altman_plot.savefig(
                f"{base_path}/{identity}_bland_altman_plot.png", dpi=300
            )
            plt.close(self.bland_altman_plot)
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
        sample_data = pd.concat([
            dataset[correct_predictions & (dataset.y_test == outcome)]
            .head(n_examples_per_outcome)
            .sort_values("y_test")
            for outcome in dataset["y_test"].unique()
        ])
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
