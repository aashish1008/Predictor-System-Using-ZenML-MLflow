import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        # plots the distribution of a numerical feature using a histogram and KDE.
        plt.figure(figsize=(10, 8))
        sns.histplot(data=df, x=feature, kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        # Plots the distribution of a categorical feature using a bar plot.
        plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x=feature, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        # executes the univariate analysis using current strategy.
        self._strategy.analyze(df, feature)

if __name__ == "__main__":
    df = pd.read_csv("extracted_data/attrition_dataset.csv")
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'DailyRate')

    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'BusinessTravel')