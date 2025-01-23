from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandera import DataFrameSchema


class DataReaderInterface(ABC):
    schema: Union[DataFrameSchema, str, Path, None]

    @abstractmethod
    def read(self, n_rows: Optional[int]) -> DataFrame: ...


class PreprocessorInterface(ABC):
    @abstractmethod
    def fit_transform(self, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]: ...

    @abstractmethod
    def transform(self, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]: ...


class DataServiceInterface(ABC):
    data_reader: DataReaderInterface
    target: str
    features: List[str] | None
    test_size: float
    random_state: int
    X_train: DataFrame = field(default_factory=DataFrame, init=False)
    X_test: DataFrame = field(default_factory=DataFrame, init=False)
    y_train: Series = field(default_factory=Series, init=False)
    y_test: Series = field(default_factory=Series, init=False)
    test_index: list = field(default_factory=list, init=False)
    train_index: list = field(default_factory=list, init=False)

    @abstractmethod
    def read(self): ...

    @abstractmethod
    def load(self): ...
