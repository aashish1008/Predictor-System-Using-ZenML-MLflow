from zenml import step
import logging
from typing_extensions import Annotated
from zenml import ArtifactConfig, save_artifact, step
from zenml.enums import ArtifactType
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import step, Model
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="attrition_predictor",
    version=None,
    license="Apache 2.0",
    description="Employee Attrition prediction model"
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "logistic") -> Annotated[
    Pipeline,
    ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)
]:
    # Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    # identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical Columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    global model
    if model_name == "logistic":
        model = LogisticRegression()

    # Define the model training pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # start an MLflow ruu to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one activa

    try:
        # enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the Logistic Regression Model.")
        pipeline.fit(X_train, y_train)

        logging.info("Model training completed.")

        # Log the columns that the model experts
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )

        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )

        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # end the MLflow run
        mlflow.end_run()

    save_artifact(pipeline, name="model", artifact_type=ArtifactType.MODEL)

    return pipeline
