from pydantic import BaseModel, field_validator


class ProductInfo(BaseModel, extra="forbid"):
    """The product information.

    Attributes:
    -----------
    product_key: str
    supplier: str
    hierarchy_level_1
    hierarchy_level_2
    di_or_dom
    seasonal
    cat_edition
    spring_summer
    weeks_out
    status
    sale_price_inc_vat
    forecast_per_week
    actuals_per_week
    """

    title: str
    text: str

    def __str__(self):
        return f"{self.title}:\n\n{self.text}"


class DiscontinuedTFOutput(BaseModel, extra="forbid"):
    """Prediction for the discontinuation of a product.
    It is a binary classification output that takes the value 0 if the product is not
    discontinued and 1 if it is discontinued.

    Attributes:
    -----------
    DiscontinuedTF: int
        A binary classification output that takes the value 0 if the product is not
    discontinued and 1 if it is discontinued.
    """

    discontinued_tf: int

    @field_validator("discontinued_tf")
    def validate_rating(cls, discontinued_tf):
        if discontinued_tf not in [0, 1]:
            raise ValueError("The value must be 0 or 1.")
        return discontinued_tf
