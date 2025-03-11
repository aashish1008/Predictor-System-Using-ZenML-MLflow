import pandas as pd
from src.feature_engineering import (
    LogTransformation,
    StandardScaling,
    MinMaxScaling,
    OneHotEncoding,
    FeatureEngineering
)
from zenml import step


@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = "log", features: list = None) -> pd.DataFrame:
    # performs feature engineering using FeatureEngineer and selected strategy.
    # ensure features is a list, even if not provided
    if features is None:
        features = []
    if strategy == "log":
        transformed = FeatureEngineering(LogTransformation(features=features))
    elif strategy == "standard_scaling":
        transformed = FeatureEngineering(StandardScaling(features=features))
    elif strategy == "minmax_scaling":
        transformed = FeatureEngineering(MinMaxScaling(features=features))
    elif strategy == "onehot_encoding":
        transformed = FeatureEngineering(OneHotEncoding(features=features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")
    transformed_df = transformed.apply_feature_engineering(df)
    return transformed_df
