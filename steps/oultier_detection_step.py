import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection

from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame, strategy: str, method: str = "remove") -> pd.DataFrame:
    # Detects and removes outlier using Outlier Detector.

    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame...")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")

    df_numeric = df.select_dtypes(include=[int, float])
    global outlier_detector
    if strategy == "zscore":
        outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    elif strategy == "IQR":
        outlier_detector = OutlierDetector(IQROutlierDetection())
    else:
        logging.error(f"Choose the correct strategy zscore or IQR. You choose : {strategy}")

    outliers = outlier_detector.detect_outliers(df_numeric)

    df_cleaned = outlier_detector.handle_outliers(df_numeric, method=method)

    return df_cleaned
