import pandas as pd
import os
from src.ingest_data import DataIngestionFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    # Ingest data from a ZIP file using the appropriate DataIngestor.
    file_path = "data/dataset.zip"
    file_extension = os.path.splitext(file_path)[1]
    # Get the appropriate DataIngestor
    data_ingestor = DataIngestionFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path=file_path, priority_format="csv")
    return df
