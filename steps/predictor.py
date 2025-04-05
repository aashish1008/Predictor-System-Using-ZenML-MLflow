import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
        service: MLFlowDeploymentService,
        input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service."""

    # start the service
    service.start(timeout=10)

    # load the input data from JSON string
    data = json.loads(input_data)

    # extract the actual data and expected columns
    data.pop("columns", None)  # rwmove columns if it's present
    data.pop("index", None)  # remove index if it's present

    # define the columns the model expects
    expected_columns = [
        "Age",
        "Attrition",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    # convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # run the prediction
    prediction = service.predict(data_array)

    return prediction
