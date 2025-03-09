import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod


# Using Strategy Design Pattern for bivariate analysis
# Subclasses must implement the analyze method
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass


# concrete strategy for Numerical vs Numerical Analysis
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        # plots the relationship between two numerical features using a scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# concrete strategy for Categorical vs Numerical Analysis
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        # plots the relationship between a categorical feature and a numerical feature using a box plot.
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=df, x=feature1, y=feature2)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# context class that uses a BivariateAnalysisStrategy
# this class allows you to switch between different bivariate analysis strategies
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        # executes the bivariate analysis using the current strategy
        self._strategy.analyze(df, feature1, feature2)


if __name__ == "__main__":
    df = pd.read_csv("extracted_data/attrition_dataset.csv")
    analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    analyzer.execute_analysis(df, 'PerformanceRating', 'Age')

    analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    analyzer.execute_analysis(df, 'Attrition', 'Age')
