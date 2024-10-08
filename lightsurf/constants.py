import numpy as np

WAVELENGTHS = 10.**(4.179 + 6.E-6 * np.arange(8575))
A = (5.792105 * 10 ** -2) / (238.0185 - (10 ** 4 / WAVELENGTHS) ** 2)
B = (1.67917 * 10 ** -3) / (57.362 - (10 ** 4 / WAVELENGTHS) ** 2)
WAVELENGTH_AIR = WAVELENGTHS / (1.00 + A + B)
