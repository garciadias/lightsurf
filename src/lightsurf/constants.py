import numpy as np

APOGEE_SPECTRUM_LENGTH = 8576
WAVELENGTHS = 10.0 ** (4.179 + 6.0e-6 * np.arange(APOGEE_SPECTRUM_LENGTH))
A = (5.792105 * 10**-2) / (238.0185 - (10**4 / WAVELENGTHS) ** 2)
B = (1.67917 * 10**-3) / (57.362 - (10**4 / WAVELENGTHS) ** 2)
APOGEE_WAVELENGTH_AIR = WAVELENGTHS / (1.00 + A + B)
APOGEE_WAVELENGTH_AIR = APOGEE_WAVELENGTH_AIR[:-1]
APOGEE_WAVELENGTH_AIR_STR = [f"{wave:.2f}" for wave in APOGEE_WAVELENGTH_AIR]

APOGEE_ABUNDANCE_TARGETS = [
    "FE_H",
    "C_FE",
    "CA_FE",
    "K_FE",
    "MG_FE",
    "NI_FE",
    "O_FE",
    "SI_FE",
    "TI_FE",
    # "AL_FE",
    # "CR_FE",
    # "MN_FE",
    # "NA_FE",
    # "V_FE",
]
APOGEE_PARAMETERS = ["TEFF", "LOGG"] + APOGEE_ABUNDANCE_TARGETS

COLORS = {
    "red": "#ee1c2e",
    "blue": "#00539f",
    "yellow": "#fcd700",
    "green": "#008800",
    "light_blue": "#007eb3",
    "orange": "#f97b02",
    "purple": "#991384",
    "black": "#000000",
    "white": "#ffffff",
    "grey": "#808080",
}
