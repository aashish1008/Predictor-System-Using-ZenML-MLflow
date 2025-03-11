import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelBuildingStrategy(ABC):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        pass


class LogisticRegressionModel(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series")

        logging.info("Initializing Logistic Regression model with scaling")

        # creating a pipeline with standard Scaling and logistic regression
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # feature scaling
                ("model", LogisticRegression()),  # Logistic Regression Model
            ]
        )

        logging.info("Training Logistic Regression Model.")
        pipeline.fit(X_train, y_train)

        logging.info("Model training completed")
        return pipeline


class KNNModel(ModelBuildingStrategy):
    def __init__(self):
        pass

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        pass


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)

