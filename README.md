# Lightsurf

Lightsurf is a simple a repository of machine learning models applications for stellar spectroscopy.

## Requirements

- Python 3.12
- poetry

## Installation

```bash
poetry install
```

## Loading shell

Loading your virtual environment on poetry is done by running:

```bash
poetry shell
```

## Usage

### Downloading spectral data

To download the spectral data for M dwarfs, run:

```bash
task download
```

### Preprocessing the data

To combine all downloaded `fits` files into a single `csv` file, run:

```bash
task combine
```

This command will combine all the `fits` files into a single `csv` file that will be saved at the folder `data/raw_data/flux_abundances`.

The file will contain the columns listed on the variable
`lightsurf.constants.APOGEE_WAVELENGTH_AIR_STR` and the abundances at `lightsurf.constants.APOGEE_PARAMETERS`.

### Training the model

To train the model, run:

```bash
task train
```

This task allows you to choose the model to be trained and the number of rows to be used for training.

For example, to train the model `cnn_lstm` with 1000 rows, run:

```bash
task train --model cnn_lstm --rows 1000
```

### MLflow

All the experiments are logged using MLflow. To visualize the experiments, run:

```bash
task mlflow
```

I recommend opening a new terminal for this command. This will start the MLflow server and you can access it at `http://localhost:5000`.

## Testing

The repository uses `pytest` for testing. To run the tests, run:

```bash
task test
```

Other auxiliary commands can be found by running:

```bash
task --list
```
