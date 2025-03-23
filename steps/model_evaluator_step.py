import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluation
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
        trained_model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> Tuple[dict, float]:
    logging.info("Applying the same preprocessing to the test data.")

    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    evaluator = ModelEvaluator(ClassificationModelEvaluation())

    evaluation_metrics = evaluator.evaluate(trained_model.named_steps["models"], X_test_processed, y_test)

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary..")

    accuracy = evaluation_metrics.get("Accuracy Score")
    return evaluation_metrics, accuracy
