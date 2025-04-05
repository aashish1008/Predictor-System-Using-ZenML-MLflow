import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports sample attrition data for testing the model."""
    data = {
        "Age": [35, 42],
        "Attrition": ["Yes", "No"],
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently"],
        "DailyRate": [1100, 900],
        "Department": ["Sales", "Research & Development"],
        "DistanceFromHome": [5, 12],
        "Education": [3, 4],
        "EducationField": ["Life Sciences", "Medical"],
        "EnvironmentSatisfaction": [4, 2],
        "Gender": ["Male", "Female"],
        "HourlyRate": [60, 45],
        "JobInvolvement": [3, 2],
        "JobLevel": [2, 1],
        "JobRole": ["Sales Executive", "Research Scientist"],
        "JobSatisfaction": [4, 3],
        "MaritalStatus": ["Married", "Single"],
        "MonthlyIncome": [5000, 4200],
        "MonthlyRate": [20000, 19000],
        "NumCompaniesWorked": [1, 3],
        "OverTime": ["Yes", "No"],
        "PercentSalaryHike": [14, 11],
        "PerformanceRating": [3, 4],
        "RelationshipSatisfaction": [3, 2],
        "StockOptionLevel": [1, 0],
        "TotalWorkingYears": [10, 8],
        "TrainingTimesLastYear": [2, 3],
        "WorkLifeBalance": [3, 2],
        "YearsAtCompany": [5, 4],
        "YearsInCurrentRole": [3, 2],
        "YearsSinceLastPromotion": [1, 2],
        "YearsWithCurrManager": [3, 2],
    }

    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")
    return json_data
