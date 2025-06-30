from pathlib import Path
from typing import Optional

import mlflow
import pandera as pa
from sklearn.model_selection import RandomizedSearchCV

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.controllers.controller import create_model_controller
from lightsurf.domain.interfaces.schemas.apogee_spectrum import SCHEMA_DICT
from lightsurf.domain.models.deep_models import CnnLstmAttentionModel, LSTMRegressor
from lightsurf.domain.services.data.data_service import FileDataReader

global random_state
random_state = 1990


def run_deep_experiment(
    model,
    input_path: str,
    model_path: str,
    schema: str | Path | pa.DataFrameSchema | None = None,
    target_variable: str = "FE_H",
    run_name="Deep Experiment",
    n_rows: Optional[int] = 100,
):
    data_reader = FileDataReader(
        input_path,
        schema=schema,
    )

    params = {
        "data_reader": data_reader,
        "target_variable": target_variable,
        "features": APOGEE_WAVELENGTH_AIR_STR + [target_variable],
        "model": model,
        "model_path": model_path,
        "test_size": 0.2,
        "random_state": random_state,
        "n_rows": n_rows,
        "feature_selection": None,
        "output_filetype": "pkl",
    }
    mlflow.set_experiment("lightsurf")
    with mlflow.start_run(run_name=run_name):
        controller = create_model_controller(**params)
        # mlflow.log_params(params)
        controller.train()
        model = controller.model

        mlflow.sklearn.log_model(model, "model")
        metrics = controller.service.model_evaluator.model_evaluation.model_dump()
        mlflow.log_metrics(metrics)
        if hasattr(controller.service.model_trainer.model, "best_estimator_"):
            best_estimator = controller.service.model_trainer.model.best_estimator_
            mlflow.tensorflow.log_model(best_estimator.model, "best_estimator")
            best_params = controller.service.model_trainer.model.best_params_
            if "features" in best_params:
                del best_params["features"]
            mlflow.log_params({"best_params": best_params})
        else:
            mlflow.tensorflow.log_model(
                controller.service.model_trainer.model.model, "Model"
            )

        mlflow.log_artifacts(model_path)
    return controller


def train_lstm_regressor(
    n_rows=1000,
    input_path="data/raw_data/flux.csv",
    schema: str | Path | pa.DataFrameSchema | None = None,
    target_variable="FE_H",
):
    module_path = Path(__file__).parents[4]
    input_path = f"{module_path}/{input_path}"
    if isinstance(schema, str):
        if schema in SCHEMA_DICT:
            schema = SCHEMA_DICT.get(schema)
    model_path = f"{module_path}/models/deep_models/"
    checkpoint_path = Path(f"{module_path}/models/deep_models/")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file_path = f"{checkpoint_path}/lstm_regressor.keras"
    lstm_regressor = LSTMRegressor(
        checkpoint_path=checkpoint_file_path,
    )
    model = RandomizedSearchCV(
        lstm_regressor,
        param_distributions={
            "lstm_units": [16, 32, 64, 128],
            "dense_units": [16, 32, 64, 128],
            "dropout": [0.1, 0.15, 0.2, 0.25],
            "epochs": [50],
            "batch_size": [64, 128],
            "verbose": [1],
            "random_state": [random_state],
        },
        n_iter=1000,
        n_jobs=1,
        cv=2,
        random_state=random_state,
        return_train_score=True,
    )
    return run_deep_experiment(
        model,
        run_name="LSTM",
        n_rows=n_rows,
        input_path=input_path,
        model_path=model_path,
        schema=schema,
        target_variable=target_variable,
    )


def train_cnn_lstm_regressor(
    n_rows=1000,
    input_path="data/raw_data/flux.csv",
    schema: str | Path | pa.DataFrameSchema | None = None,
    target_variable="FE_H",
):
    module_path = Path(__file__).parents[4]
    input_path = f"{module_path}/{input_path}"
    if isinstance(schema, str):
        if schema in SCHEMA_DICT:
            schema = SCHEMA_DICT.get(schema)
    model_path = f"{module_path}/models/deep_models/"
    checkpoint_path = Path(f"{module_path}/models/deep_models/")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file_path = f"{checkpoint_path}/lstm_regressor.keras"
    lstm_regressor = CnnLstmAttentionModel(
        checkpoint_path=checkpoint_file_path,
    )
    model = RandomizedSearchCV(
        lstm_regressor,
        param_distributions={
            "epochs": [50],
            "batch_size": [64, 128, 256],
            "verbose": [1],
            "random_state": [random_state],
            "cnn_kernel_size": [3, 5, 7],
            "cnn_strides": [1, 2, 3],
            "learning_rate": [0.001, 0.01, 0.1],
            "dropout": [0.1, 0.15, 0.2, 0.25],
            "lstm_units": [16, 32, 64, 128, 256, 512],
            "cnn_filters": [
                16,
                32,
                64,
                128,
                256,
                512,
                [16, 32, 64, 128, 256, 512],
                [len(APOGEE_WAVELENGTH_AIR_STR) // i for i in [1, 2, 4, 8]],
            ],
            "dense_units": [
                16,
                32,
                64,
                128,
                256,
                512,
                [16, 32, 64, 128, 256, 512],
                [20, 8],
                [16, 32, 64],
                [16, 32, 64, 128],
            ],
        },
        n_iter=1000,
        n_jobs=1,
        cv=2,
        random_state=random_state,
    )
    return run_deep_experiment(
        model,
        run_name="CNN_LSTM",
        n_rows=n_rows,
        input_path=input_path,
        model_path=model_path,
        schema=schema,
        target_variable=target_variable,
    )
