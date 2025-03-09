import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# subclasses must implement the apply_transformation method
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initializes the LogTransformation with the specific features to transform.
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed")
        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Applies standard scaling to the specified features in the DataFrame.
        logging.info(f"Applying Standard Scaler to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Applies Min-Max scaling to the specified features in the DataFrame.
        logging.info(f"Applying Min-Max Scaler to features: {self.features} with range {(0, 1)}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("MinMax scaling completed.")
        return df_transformed


class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(drop="first", sparse_output=False)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Applies one-hot encoding to the specified categorical features in the DataFrame.
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("one-hot encoding completed")
        return df_transformed


class FeatureEngineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


if __name__ == "__main__":
    df = pd.read_csv('extracted_data/attrition_dataset.csv')
    transformed = FeatureEngineering(LogTransformation(features=['Age', 'DailyRate']))
    # Log Transformation Example
    df_log_transformed = transformed.apply_feature_engineering(df)
    print(df_log_transformed)

    # Standard Scaling Example
    transformed.set_strategy(StandardScaling(features=['Age', 'DailyRate']))
    df_standard_scaled = transformed.apply_feature_engineering(df)
    print(df_standard_scaled)

    # Min-Max Scaling Example
    transformed.set_strategy(MinMaxScaling(features=['Age', 'DailyRate']))
    df_minmax_scaled = transformed.apply_feature_engineering(df)
    print(df_minmax_scaled)

    # One-Hot Encoding Example
    transformed.set_strategy(OneHotEncoding(features=['BusinessTravel', 'MaritalStatus']))
    df_onehot_encoded = transformed.apply_feature_engineering(df)
    print(df_onehot_encoded)
