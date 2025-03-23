import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# abstract class
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        pass


# concrete strategy for Regression Model Evaluation
class RegressionModelEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # evaluating a classification model using accuracy, precision, recall, f1 score
        logging.info("Predicting using the trained model...")
        y_pred = model.predict(X_test)

        assert y_pred.shape == y_test.shape, f"Shape mismatch: y_pred has shape {y_pred.shape}, but y_test has shape {y_test.shape}. Ensure both have the same dimensions."

        logging.info("Calculating evaluation metrics..")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Mean Squared Error": mse,
            "R-Sqaured": r2
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")

        return metrics


# concrete strategy for Classification Model Evaluation
class ClassificationModelEvaluation(ModelEvaluationStrategy):
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # evaluating a classification model using accuracy, precision, recall, f1 score
        logging.info("Classifying using the trained model...")
        y_pred = model.predict(X_test)

        assert y_pred.shape == y_test.shape, f"Shape mismatch: y_pred has shape {y_pred.shape}, but y_test has shape {y_test.shape}. Ensure both have the same dimensions."

        logging.info("Calculating evaluation metrics..")
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score = precision_recall_fscore_support(y_test, y_pred)

        metrics = {
            "Accuracy Score": accuracy,
            "Precision Score": precision,
            "Recall Score": recall,
            "F1 Score": f1_score
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")

        return metrics


# Context class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model:Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Evaluating the model using the selected strategy..")
        return self._strategy.evaluate_model(model, X_test, y_test)



