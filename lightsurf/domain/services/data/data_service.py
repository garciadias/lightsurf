from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple, Union

import pandas as pd
import pandera as pa
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split

from lightsurf.domain.interfaces.data_reader import (
    DataReaderInterface,
    DataServiceInterface,
)


@dataclass
class FileDataReader(DataReaderInterface):
    path: Union[Path, str]
    schema: Union[pa.DataFrameSchema, str, Path, None]
    read_function: Callable = pd.read_csv

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if isinstance(self.schema, str):
            self.schema = Path(self.schema)
        if isinstance(self.schema, Path):
            if not self.schema.exists():
                raise FileNotFoundError(f"Schema file {self.schema} not found")

    def read(self, n_rows: int | None = None) -> DataFrame:
        df = self.read_function(self.path, nrows=n_rows)
        if isinstance(self.schema, str) or isinstance(self.schema, Path):
            with open(self.schema, "r") as file:
                schema_yaml = file.read()
            self.schema = pa.DataFrameSchema.from_yaml(schema_yaml)
        if self.schema is not None:
            df = self.schema(df)
        if "id" in df.columns:
            df.set_index("id", inplace=True)
        return df


@dataclass
class DataService(DataServiceInterface):
    data_reader: DataReaderInterface
    target: str
    X_train: DataFrame = field(default_factory=DataFrame, init=False)
    X_test: DataFrame = field(default_factory=DataFrame, init=False)
    y_train: Series = field(default_factory=Series, init=False)
    y_test: Series = field(default_factory=Series, init=False)
    features: List[str] | None = None
    test_size: float = 0.2
    random_state: int = 42
    n_rows: int | None = None
    stratify: bool = False

    def read(self):
        data = self.data_reader.read(self.n_rows)
        return data

    def get_features(self, data: DataFrame):
        self.features = (
            self.features if self.features is not None else list(data.columns)
        )
        self.features.remove(self.target)
        return self.features

    def train_test_split(
        self, data: DataFrame, split_by: str | None = None
    ) -> Tuple[DataFrame, DataFrame, Series, Series]:
        features = self.get_features(data)
        print(f"🧮 Data loaded with shape: {data.shape}")
        # Drop rows with null target values
        data = data.dropna(subset=[self.target])
        print(f"🧮 Data loaded with shape after dropping null values: {data.shape}")
        X = data[features]
        y = data[self.target]
        if split_by is not None:
            unique_values = pd.Series(data[split_by].unique())
            test_indexes = unique_values.sample(
                frac=self.test_size, random_state=self.random_state
            )
            X_train = X[~X[split_by].isin(test_indexes)]
            X_test = X[X[split_by].isin(test_indexes)]
            y_train = y[~X[split_by].isin(test_indexes)]
            y_test = y[X[split_by].isin(test_indexes)]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.stratify else None,
            )
        return X_train, X_test, y_train, y_test

    def load(
        self,
        split_by: str | None = None,
    ):
        data = self.read()
        X_train, X_test, y_train, y_test = self.train_test_split(data, split_by)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_index = self.X_train.index.to_list()
        self.test_index = self.X_test.index.to_list()
