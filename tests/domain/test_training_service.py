import pytest

from lightsurf.domain.services.training_service import FileModelRepository


def test_file_model_repository_filetype(tmp_path):
    with pytest.raises(AttributeError):
        FileModelRepository(base_path=tmp_path, filetype="wrong")
