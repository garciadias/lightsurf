import pytest

from lightsurf.domain.services.data.data_service import FileDataReader


@pytest.fixture
def testing_data_reader(testing_data_path, test_schema):
    return FileDataReader(path=testing_data_path, schema=test_schema)
