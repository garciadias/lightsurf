from typing import List, Union

from pydantic import BaseModel


class ModelEvaluationClassification(BaseModel):
    accuracy: Union[float, List[float]]
    precision: Union[float, List[float]]
    recall: Union[float, List[float]]
    f1_score: Union[float, List[float]]

    def __str__(self):
        results = self.model_dump()
        return ", ".join(
            [
                f"{key.replace("_", " ").title()}: {value:0.3f}"
                for key, value in results.items()
            ]
        )

    def model_dump(self):
        if any(isinstance(i, list) for i in self.__dict__.values()):
            unnested_dict = {}
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    for i, v in enumerate(value, start=1):
                        unnested_dict[f"{key}_{i}"] = v
                else:
                    unnested_dict[key] = value
            return unnested_dict
        else:
            return self.__dict__


class ModelEvaluationRegression(BaseModel):
    mean_absolute_error: Union[float, List[float]]
    mean_squared_error: Union[float, List[float]]
    r2_score: Union[float, List[float]]

    def __str__(self):
        results = self.model_dump()
        return ", ".join(
            [
                f"{key.replace("_", " ").title()}: {value:0.3f}"
                for key, value in results.items()
            ]
        )

    def model_dump(self):
        if any(isinstance(i, list) for i in self.__dict__.values()):
            unnested_dict = {}
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    for i, v in enumerate(value, start=1):
                        unnested_dict[f"{key}_{i}"] = v
                else:
                    unnested_dict[key] = value
            return unnested_dict
        else:
            return self.__dict__
