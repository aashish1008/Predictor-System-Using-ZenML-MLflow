from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.oultier_detection_step import outlier_detection_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_bulding_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # the name uniquely identifies this model
        name="attrition_predictor"
    ),
)
def ml_pipeline():
    # define an end to end machine learning pipeline

    # Data Ingestion Step
    raw_data = data_ingestion_step(file_path="./data/dataset.zip")

    # Handling Missing Values
    filled_data = handle_missing_values_step(raw_data)

    # feature engineering step
    log_transformed = feature_engineering_step(df=filled_data, strategy="log", features=["HourlyRate", "DistanceFromHome"])

    # outline detection step
    clean_data = outlier_detection_step(df=log_transformed, strategy="zscore", method="remove")

    # data splitting step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, "Attrition")

    # model building step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model evaluation step
    evaluation_metrics, accuracy = model_evaluator_step(trained_model=model, X_test=X_test, y_test=y_test)

    return model
