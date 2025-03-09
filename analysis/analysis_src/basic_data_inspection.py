import pandas as pd
from abc import ABC, abstractmethod


# Using Strategies Design Pattern for basic data inspection
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass


# concrete strategy for data types inspection
class DataTypesInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nData Types and Non-null Counts: ")
        print(df.info())


# concrete strategy for summary statistics inspection
class SummaryStatisticsInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        # Prints summary statistics for numerical and categorical features.
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# context class that uses a DataInspectionStrategy
# This class allows you to switch between different data inspection strategies.

class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        # the strategy used for the data inspection
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        # the new strategy used for the data inspection
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)


if __name__ == "__main__":
    df = pd.read_csv("extracted_data/attrition_dataset.csv")

    inspector = DataInspector(DataTypesInspection())
    inspector.execute_inspection(df)

    inspector.set_strategy(SummaryStatisticsInspection())
    inspector.execute_inspection(df)
