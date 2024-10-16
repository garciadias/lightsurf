from pydantic import BaseModel


class Abundances(BaseModel):
    FE_H = float
    C_FE = float
    O_FE = float
    NA_FE = float
    MG_FE = float
    AL_FE = float
    SI_FE = float
    K_FE = float
    CA_FE = float
    TI_FE = float
    V_FE = float
    CR_FE = float
    MN_FE = float
    NI_FE = float
