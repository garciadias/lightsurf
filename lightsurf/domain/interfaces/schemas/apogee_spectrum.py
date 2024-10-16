import pandera as pa

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS, APOGEE_WAVELENGTH_AIR_STR

APOGEE_SCHEMA = pa.DataFrameSchema(
    columns={
        f"{variable}": pa.Column(float, nullable=True)
        for variable in APOGEE_WAVELENGTH_AIR_STR + APOGEE_ABUNDANCE_TARGETS
    },
)

SCHEMA_DICT = {"apogee": APOGEE_SCHEMA}
