from pathlib import Path

import mlflow
import pandas as pd
import pandera as pa

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.controllers.controller import (
    RANDOM_STATE,
    create_model_controller,
)
from lightsurf.domain.interfaces.schemas.apogee_spectrum import SCHEMA_DICT
from lightsurf.domain.services.data.data_service import FileDataReader
from lightsurf.domain.services.training_service import FeatureSelector, XGBoostTrainer


def run_shallow_experiment(
    model,
    target_variable: str,
    input_path: str,
    model_path: str,
    schema: str | Path | pa.DataFrameSchema | None,
    run_name="Shallow Experiment",
    n_rows=1000,
    feature_selection="select_from_model",
):
    data_reader = FileDataReader(
        input_path,
        schema=schema,
    )
    param_distributions = {
        "n_estimators": range(400, 1000, 50),
        "max_depth": [3, 5, 7, 9, 11],
        "learning_rate": [0.0001, 0.01, 0.05, 0.1, 0.3, 0.5],
        "feature_selector": [FeatureSelector(selection_type=feature_selection)],
    }
    params = {
        "data_reader": data_reader,
        "target_variable": target_variable,
        "features": APOGEE_WAVELENGTH_AIR_STR + [target_variable],
        "model": model,
        "model_path": model_path,
        "test_size": 0.2,
        "param_distributions": param_distributions,
        "random_state": RANDOM_STATE,
        "n_rows": n_rows,
        "output_filetype": "pkl",
        "feature_selection": feature_selection,
    }

    mlflow.set_experiment("lightsurf")
    with mlflow.start_run(run_name=run_name):
        controller = create_model_controller(**params)
        del params["features"]
        del params["data_reader"]
        mlflow.log_params(params)
        controller.fit()
        model = controller.model
        features = controller.service.data_service.X_train.columns.tolist()
        if feature_selection:
            selected_features = getattr(
                controller.service.feature_selector, "selected_features_", features
            )
            n_features = len(features)
            n_selected_features = len(selected_features)
            selected_features = "\n".join(selected_features)
            mlflow.log_params(
                {
                    "features": features,
                    "selected_features": selected_features,
                    "n_features": n_features,
                    "n_selected_features": n_selected_features,
                }
            )
        metrics = controller.service.model_evaluator.model_evaluation.model_dump()
        print(pd.DataFrame(metrics, index=[0]))
        mlflow.log_metrics(metrics)
        if hasattr(model, "best_params_"):
            mlflow.log_params({"best_params_": model.best_params_})
        mlflow.log_artifacts(model_path)
        del controller.service.data_service
        mlflow.sklearn.log_model(model, "model")
    return controller


def train_xgboost_regressor(
    n_rows=1000,
    input_path="data/raw_data/flux_abundances.csv",
    schema: str | Path | pa.DataFrameSchema | None = "",
    target_variable="FE_H",
):
    xgboost = XGBoostTrainer()
    if isinstance(schema, str):
        if schema in SCHEMA_DICT:
            schema = SCHEMA_DICT.get(schema)
    return run_shallow_experiment(
        model=xgboost,
        target_variable=target_variable,
        input_path=input_path,
        model_path="data/models/",
        schema=schema,
        run_name="XGBRegressor",
        n_rows=n_rows,
    )
