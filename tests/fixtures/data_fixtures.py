import tempfile
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier


class TestSchema(pa.DataFrameModel):
    # Iris dataset schema
    sepal_length: Series[float]
    sepal_width: Series[float]
    petal_length: Series[float]
    petal_width: Series[float]
    target: Series[int]


@pytest.fixture(scope="module")
def test_schema():
    return TestSchema


@pytest.fixture
def testing_model():
    return DummyClassifier()


@pytest.fixture
def iris_data():
    return load_iris(as_frame=True)


@pytest.fixture
def X_train(iris_data):
    return iris_data.data.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )


@pytest.fixture
def y_train(iris_data):
    return iris_data.target


@pytest.fixture
def df_for_testing(X_train, y_train):
    df = X_train
    df["target"] = y_train
    df.index.name = "id"
    return df


@pytest.fixture
def df_text_for_testing():
    df = pd.DataFrame(
        {
            "text": ["I am running", "You are running"],
            "title": ["Title 1", "Title 2"],
            "target": [0, 1],
        },
        index=[0, 1],
    )
    return df


@pytest.fixture
def testing_data_path(df_for_testing):
    temp_dir = tempfile.mkdtemp()
    file_stem = "test_data"
    file_out = Path(f"{temp_dir}/{file_stem}.csv")
    df_for_testing.to_csv(file_out)
    df_for_testing.to_csv("data/iris_data.csv")
    return "data/iris_data.csv"
