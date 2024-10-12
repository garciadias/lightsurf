import mlflow
import pandas as pd

from lightsurf.domain.controllers.controller import (
    RANDOM_STATE,
    create_model_controller,
)
from lightsurf.domain.services.data.data_service import (
    FileDataReader,
)


def run_shallow_experiment(
    model,
    input_path: str,
    model_path: str,
    schema_path: str,
    run_name="Shallow Experiment",
    n_rows=1000,
    feature_selection="select_from_model",
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
        "random_state": RANDOM_STATE,
        "n_rows": n_rows,
        "output_filetype": "pkl",
        "feature_selection": feature_selection,
    }

    mlflow.set_experiment("Sainsburys")
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
