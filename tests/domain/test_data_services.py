from unittest.mock import mock_open, patch

import pytest
from pandas import Series
from pandas.core.frame import DataFrame
from pandera.errors import SchemaError

from lightsurf.domain.services.data.data_service import DataService, FileDataReader


@pytest.fixture
def data_service_without_preprocessor(testing_data_path, test_schema):
    data_service = DataService(
        data_reader=FileDataReader(testing_data_path, test_schema), target="target"
    )
    return data_service


def test_file_data_reader_can_receive_path_to_schema(testing_data_path, test_schema):
    yaml_content = test_schema.to_yaml()
    data_reader = FileDataReader(testing_data_path, test_schema)
    # mock file read
    with pytest.raises(SchemaError):
        with patch("builtins.open", mock_open(read_data=yaml_content), create=True):
            data_reader.read()


def test_data_service_can_be_created(testing_data_reader):
    data_service = DataService(target="target", data_reader=testing_data_reader)
    assert data_service is not None


def test_data_service_can_read(data_service_without_preprocessor):
    data = data_service_without_preprocessor.read()
    assert isinstance(data, DataFrame)


def test_data_service_get_sequence_data(data_service_without_preprocessor):
    data_service_without_preprocessor.load()
    X_train = data_service_without_preprocessor.X_train
    y_train = data_service_without_preprocessor.y_train
    y_test = data_service_without_preprocessor.y_test
    X_test = data_service_without_preprocessor.X_test
    assert isinstance(X_train, DataFrame)
    assert isinstance(y_train, Series)
    assert isinstance(X_test, DataFrame)
    assert isinstance(y_test, Series)
    assert len(X_train) == len(y_train)
