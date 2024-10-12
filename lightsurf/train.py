import click

from lightsurf.domain.controllers.deep_controller import run_deep_experiment
from lightsurf.domain.controllers.shallow_controller import run_shallow_experiment


@click.command()
@click.option(
    "--m",
    default="lstm",
    help="Define the model to train. Options: " "lstm or xgb_classifier",
)
@click.option(
    "--n_rows",
    default=1000,
    help="Number of rows to train the model. "
    "Default: 1000 rows. If -1 is passed, all rows will be used.",
)
def train(m, n_rows):
    MODELS = {
        "lstm": run_deep_experiment,
        "xgb_classifier": run_shallow_experiment,
    }
    model = MODELS[m]
    print(f"Training {m} model with {n_rows} rows.")
    if n_rows == -1:
        model(n_rows=None)
    else:
        n_rows = int(n_rows)
        model(n_rows=n_rows)


if __name__ == "__main__":
    train()
