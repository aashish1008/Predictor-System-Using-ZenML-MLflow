import pandas as pd
from zenml import step
from src.handle_missing_values import DropMissingValues, FillMissingValues, MissingValueHandler


@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    # Handles missing values using MissingValueHandler and the specified strategy.

    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValues(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValues(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.execute_handle_missing_values(df)
    return cleaned_df
