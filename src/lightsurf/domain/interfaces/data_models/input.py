import numpy as np
from pydantic import BaseModel, field_validator

from lightsurf.constants import APOGEE_SPECTRUM_LENGTH


class Spectrum(BaseModel, arbitrary_types_allowed=True):
    intensities: np.ndarray | list[float]


class APOGEESpectrum(Spectrum):
    @field_validator("intensities")
    def validate_wavelengths(cls, value):
        if len(value) != APOGEE_SPECTRUM_LENGTH:
            raise IndexError(
                f"Length of wavelengths must be {APOGEE_SPECTRUM_LENGTH}"
                f", got {len(value)}"
            )
        return value
