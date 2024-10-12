import pytest
from pydantic_core import ValidationError

from lightsurf.domain.interfaces.data_models.input import (
    DiscontinuedTFOutput,
)


def test_DiscontinuedTF_output_only_accepts_DiscontinuedTF():
    with pytest.raises(ValidationError):
        DiscontinuedTFOutput(DiscontinuedTF=5, extra="extra")
    with pytest.raises(ValidationError):
        DiscontinuedTFOutput()
