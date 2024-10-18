import numpy as np
import pytest

from lightsurf.constants import APOGEE_SPECTRUM_LENGTH
from lightsurf.domain.interfaces.data_models.input import APOGEESpectrum, Spectrum


def test_spectrum():
    spectrum = Spectrum(
        intensities=np.array([1, 2, 3]),
    )
    assert np.array_equal(spectrum.intensities, np.array([1, 2, 3]))


def test_apogee_spectrum():
    apogee_spectrum = APOGEESpectrum(
        intensities=np.random.rand(APOGEE_SPECTRUM_LENGTH),
    )
    assert isinstance(apogee_spectrum, APOGEESpectrum)


def test_apogee_spectrum_raises_error_if_wrong_len():
    with pytest.raises(IndexError):
        APOGEESpectrum(intensities=np.random.rand(APOGEE_SPECTRUM_LENGTH + 1))
