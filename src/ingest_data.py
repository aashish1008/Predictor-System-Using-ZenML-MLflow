import os
import zipfile
import pandas as pd
from abc import ABC, abstractmethod
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Using Factory Design Pattern for Data Ingestion
# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str, priority_format: str = "csv") -> pd.DataFrame:
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str, priority_format: str = "csv") -> pd.DataFrame:
        # extracting .zip file and return the dataframe
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not .zip file.")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        extracted_files = os.listdir("extracted_data")

        # Check for each format type
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        json_files = [f for f in extracted_files if f.endswith(".json")]
        excel_files = [f for f in extracted_files if f.endswith((".xlsx", ".xls"))]
        parquet_files = [f for f in extracted_files if f.endswith(".parquet")]

        if len(csv_files) == 0:
            logging.info("No CSV file found in the extracted data.")
        if len(json_files) == 0:
            logging.info("No JSON file found in the extracted data.")
        if len(excel_files) == 0:
            logging.info("No EXCEL file found in the extracted data.")
        if len(parquet_files) == 0:
            logging.info("No PARQUET file found in the extracted data.")

        available_formats = {
            "csv": csv_files,
            "json": json_files,
            "excel": excel_files,
            "parquet": parquet_files
        }

        # Check if any formats exist
        existing_formats = [fmt for fmt, files in available_formats.items() if files]

        if not existing_formats:
            raise FileNotFoundError("No supported data files found in the zip archive.")

        logging.info(f"Available fomrats: {', '.join(existing_formats)}")

        if priority_format in available_formats:
            chosen_format = priority_format
        else:
            chosen_format = existing_formats[0]
            logging.warning(f"Priority format '{priority_format}' not found. Using '{chosen_format}' instead.")

        selected_files = available_formats[chosen_format]

        if len(selected_files) > 1:
            logging.info(
                f"Found {len(selected_files)} {selected_files[0].upper()} files. Using the first one: {selected_files[0]}")
        else:
            logging.info(f"using {selected_files[0].upper()} file : {selected_files[0]}")

        # setting the path for reading files into dataframe
        selected_file_path = os.path.join("extracted_data", selected_files[0])

        # return dataframe
        if chosen_format == "csv":
            return pd.read_csv(selected_file_path)
        elif chosen_format == "json":
            return pd.read_json(selected_file_path)
        elif chosen_format == "excel":
            return pd.read_excel(selected_file_path)
        elif chosen_format == "parquet":
            return pd.read_parquet(selected_file_path)


# Implement a Factory to create DataIngestors
class DataIngestionFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        # return the appropriate DataIngestor based on file extension
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            logging.info(f"No ingestor available for file extension: {file_extension}")


# if __name__ == "__main__":
#     file_path = "data/dataset.zip"
#     file_extension = os.path.splitext(file_path)[1]
#     data_ingestor = DataIngestionFactory.get_data_ingestor(file_extension=file_extension)
#
#     df = data_ingestor.ingest(file_path=file_path)
#
#     print(df.head)

