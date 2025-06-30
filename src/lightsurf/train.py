import click

from lightsurf.constants import APOGEE_PARAMETERS
from lightsurf.domain.controllers.deep_controller import (
    train_cnn_lstm_regressor,
    train_lstm_regressor,
)
from lightsurf.domain.controllers.shallow_controller import train_xgboost_regressor


@click.command()
@click.option(
    "--m",
    default="cnn_lstm",
    help="Define the model to train. Options: cnn_lstm or xgb",
)
@click.option(
    "--n_rows",
    default=1000,
    help="Number of rows to train the model. "
    "Default: 1000 rows. If -1 is passed, all rows will be used.",
)
@click.option(
    "--input_path",
    default="data/raw_data/flux_abundances.csv",
    help="Path to the input data.",
)
@click.option(
    "--schema",
    default="apogee",
    help="Path to yaml schema or str in the list ['apogee']",
)
@click.option(
    "--target_variable",
    default="FE_H",
    help="Target variable to predict. Default: FE_H, the valid options are %s"
    % APOGEE_PARAMETERS,
)
def train(m, n_rows, input_path, schema, target_variable):
    MODELS = {
        "lstm": train_lstm_regressor,
        "cnn_lstm": train_cnn_lstm_regressor,
        "xgb": train_xgboost_regressor,
    }
    model = MODELS[m]
    print(f"Training {m} model with {n_rows} rows.")
    if n_rows == -1:
        model(
            n_rows=None,
            input_path=input_path,
            schema=schema,
            target_variable=target_variable,
        )
    else:
        n_rows = int(n_rows)
        model(
            n_rows=n_rows,
            input_path=input_path,
            schema=schema,
            target_variable=target_variable,
        )


if __name__ == "__main__":
    train()
