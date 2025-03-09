import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod


class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        # Generates and displays a correlation heatmap for the numerical features in the dataframe.
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        # Generates and displays a pair plot for the selected features in the dataframe.
        sns.pairplot(df)
        plt.suptitle("Pair plot for the selected features", y=1.02)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("extracted_data/attrition_dataset.csv")
    multivariate_analyzer = SimpleMultivariateAnalysis()
    selected_features = df[
        ['Age', 'DailyRate', 'Education', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany',
         'YearsInCurrentRole']]
    multivariate_analyzer.analyze(selected_features)
