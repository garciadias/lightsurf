from pathlib import Path
from typing import Optional

import mlflow
from sklearn.model_selection import RandomizedSearchCV

from lightsurf.domain.controllers.controller import create_model_controller
from lightsurf.domain.models.deep_models import LSTMRegressor
from lightsurf.domain.services.data.data_service import (
    FileDataReader,
)

global random_state
random_state = 31869


def run_deep_experiment(
    model,
    input_path: str,
    model_path: str,
    schema_path: str,
    run_name="Deep Experiment",
    n_rows: Optional[int] = 100,
):
    data_reader = FileDataReader(
        input_path,
        schema=schema_path,
    )

    params = {
        "data_reader": data_reader,
        "target_variable": "DiscontinuedTF",
        "model": model,
        "model_path": model_path,
        "test_size": 0.2,
        "random_state": random_state,
        "n_rows": n_rows,
        "feature_selection": None,
        "output_filetype": "pkl",
        "split_by": "ProductKey",
        "sequence_split_by": ["CatEdition"],
    }
    mlflow.set_experiment("lightsurf")
    with mlflow.start_run(run_name=run_name):
        controller = create_model_controller(**params)
        mlflow.log_params(params)
        controller.train()
        model = controller.model

        mlflow.sklearn.log_model(model, "model")
        metrics = controller.service.model_evaluator.model_evaluation.model_dump()
        mlflow.log_metrics(metrics)
        best_estimator = controller.service.model_trainer.model.best_estimator_
        mlflow.tensorflow.log_model(best_estimator.model, "best_estimator")
        best_params = controller.service.model_trainer.model.best_params_
        mlflow.log_params({"best_params": best_params})

        mlflow.log_artifacts(model_path)
    return controller


def train_lstm_classifier(n_rows=1000):
    module_path = Path(__file__).parents[1]
    input_path = f"{module_path}/data/merged_data.csv"
    schemas_path = f"{module_path}/lightsurf/domain/interfaces/schemas/"
    schema_path = f"{schemas_path}/merged_data.yaml"
    model_path = f"{module_path}/data/deep_models/"
    checkpoint_path = Path(f"{module_path}/data/deep_models/")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_path}/lstm_regressor.keras"
    lstm_regressor = LSTMRegressor(
        checkpoint_path,
    )
    model = RandomizedSearchCV(
        lstm_regressor,
        param_distributions={
            "lstm_units": [150],
            "dense_units": [192],
            "dropout": [0.25],
            "epochs": [100],
            "batch_size": [200],
            "verbose": [1],
            "random_state": [random_state],
        },
        n_iter=1,
        n_jobs=1,
        cv=2,
        random_state=random_state,
    )
    return run_deep_experiment(
        model,
        run_name="LSTM",
        n_rows=n_rows,
        input_path=input_path,
        model_path=model_path,
        schema_path=schema_path,
    )
