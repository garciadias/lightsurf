from typing import List, Union

from pydantic import BaseModel


class ModelEvaluationClassification(BaseModel):
    accuracy: Union[float, List[float]]
    precision: Union[float, List[float]]
    recall: Union[float, List[float]]
    f1: Union[float, List[float]]

    def __str__(self):
        return (
            f"Accuracy: {self.accuracy:0.3f}, "
            f"Precision: {self.precision:0.3f}, "
            f"Recall: {self.recall:0.3f}, "
            f"F1: {self.f1:0.3f}, "
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
        return (
            f"mean_absolute_error: {self.mean_absolute_error:0.3f}, "
            f"mean_square_error: {self.mean_squared_error:0.3f}, "
            f"r2_score: {self.r2_score:0.3f}, "
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
