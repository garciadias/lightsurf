from pathlib import Path

import mlflow
import pandas as pd
import pandera as pa
from xgboost import XGBRegressor

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.controllers.controller import (
    RANDOM_STATE,
    create_model_controller,
)
from lightsurf.domain.interfaces.schemas.apogee_spectrum import SCHEMA_DICT
from lightsurf.domain.services.data.data_service import (
    FileDataReader,
)


def run_shallow_experiment(
    model,
    target_variable: str,
    input_path: str,
    model_path: str,
    schema: str | Path | pa.DataFrameSchema,
    run_name="Shallow Experiment",
    n_rows=1000,
    feature_selection="select_from_model",
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
        "random_state": RANDOM_STATE,
        "n_rows": n_rows,
        "output_filetype": "pkl",
        "feature_selection": feature_selection,
    }

    mlflow.set_experiment("lightsurf")
    with mlflow.start_run(run_name=run_name):
        controller = create_model_controller(**params)
        mlflow.log_params(params)
        controller.train()
        model = controller.model
        features = controller.service.data_service.X_train.columns.tolist()
        if feature_selection:
            selected_features = controller.service.feature_selector.selected_features_
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

        mlflow.log_artifacts(model_path)
        del controller.service.data_service
        mlflow.sklearn.log_model(model, "model")
    return controller


def train_xgboost_regressor(
    n_rows=1000,
    input_path="data/raw_data/flux_abundances.csv",
    schema: str | Path = "",
    target_variable="FE_H",
):
    model = XGBRegressor(random_state=RANDOM_STATE)
    if schema in SCHEMA_DICT:
        schema = SCHEMA_DICT[schema]
    return run_shallow_experiment(
        model=model,
        target_variable=target_variable,
        input_path=input_path,
        model_path="data/models/",
        schema=schema,
        run_name="XGBRegressor",
        n_rows=n_rows,
    )
