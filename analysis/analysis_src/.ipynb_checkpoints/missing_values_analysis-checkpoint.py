import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from abc import ABC, abstractmethod


# using Template Design Pattern for missing value analysis

# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValueAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        # Performs a complete missing values analysis by identifying and visualizing missing values.
        self.identity_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identity_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValueAnalysisTemplate):
    def identity_missing_values(self, df: pd.DataFrame):
        # Prints the count of missing values for each column in the dataframe.
        print("\nMissing values count by column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        # Creates a heatmap to visualize the missing values in the dataframe
        print("\nVisualizing Missing Values")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("extracted_data/attrition_dataset.csv")
    missing_values_analysis = SimpleMissingValuesAnalysis()
    missing_values_analysis.analyze(df)
