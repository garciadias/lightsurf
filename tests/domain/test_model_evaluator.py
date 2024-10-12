import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from lightsurf.domain.interfaces.data_models.evaluation import ModelEvaluationRegression
from lightsurf.domain.services.training_service import ModelEvaluator


@pytest.fixture
def tmp_path():
    base_path = Path("/tmp/test/")
    yield base_path
    if base_path.exists():
        shutil.rmtree(base_path)


def test_model_evaluator():
    with patch(
        "lightsurf.domain.services.training_service.ModelEvaluator.save_evaluation"
    ) as mock_save_evaluation:
        model_evaluator = ModelEvaluator()
        base_path = "data/evaluations"
        Path(base_path).mkdir(parents=True, exist_ok=True)
        model_evaluator.save_evaluation(identity="xgboost", base_path=base_path)
        mock_save_evaluation.assert_called_once_with(
            identity="xgboost", base_path=base_path
        )


def test_model_evaluator_creates_folder_if_does_not_exist():
    with patch(
        "lightsurf.domain.services.training_service.Path.exists"
    ) as mock_path_exists:
        mock_path_exists.return_value = False
        with patch(
            "lightsurf.domain.services.training_service.Path.mkdir"
        ) as mock_path_mkdir:
            model_evaluator = ModelEvaluator()
            model_evaluator.model_evaluation = ModelEvaluationRegression(
                mean_absolute_error=0.5,
                mean_squared_error=0.5,
                r2_score=0.5,
            )
            model_evaluator.confusion_matrix = plt.figure()
            model_evaluator.indexes = pd.DataFrame([1, 2, 3])
            model_evaluator.save_evaluation(
                identity="xgboost", base_path="data/evaluations"
            )
            mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_model_evaluation_str_method():
    str_result = str(
        ModelEvaluationRegression(
            mean_absolute_error=0.5,
            mean_squared_error=0.5,
            r2_score=0.5,
        )
    )
    assert isinstance(str_result, str)
