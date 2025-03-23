# Predictor System Using ZenML and MLflow

This project demonstrates the implementation of a machine learning pipeline using [ZenML](https://zenml.io/) and [MLflow](https://mlflow.org/). ZenML is an extensible MLOps framework for creating reproducible, production-ready machine learning pipelines, while MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. By integrating ZenML with MLflow, this project showcases how to track experiments, manage models, and streamline the deployment process.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The repository is organized as follows:

- **analysis/**: Contains analysis scripts and notebooks.
- **data/**: Directory for storing raw and processed data.
- **extracted_data/**: Contains data extracted from various sources.
- **pipelines/**: Defines the ZenML pipelines.
- **src/**: Source code for data processing, model training, and evaluation.
- **steps/**: Individual pipeline steps such as data ingestion, preprocessing, training, and evaluation.
- **config.yaml**: Configuration file for setting up the pipeline parameters.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **run_pipeline.py**: Script to execute the ZenML pipeline.

## Setup and Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aashish1008/Predictor-System-Using-ZenML-MLflow.git
   cd Predictor-System-Using-ZenML-MLflow
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize ZenML:**

   ```bash
   zenml init
   ```

5. **Install ZenML integrations:**

   This project utilizes the MLflow integration for experiment tracking. Install the necessary integration by running:

   ```bash
   zenml integration install mlflow -y
   ```

6. **Register and set up the ZenML stack:**

   Register the MLflow experiment tracker and create a ZenML stack:

   ```bash
   # Register the MLflow experiment tracker
   zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

   # Register the stack with the MLflow experiment tracker
   zenml stack register local_stack -e mlflow_experiment_tracker

   # Set the active stack to the newly created stack
   zenml stack set local_stack
   ```

   For more detailed information on configuring the MLflow experiment tracker, refer to the ZenML documentation. ([docs.zenml.io](https://docs.zenml.io/stack-components/experiment-trackers/mlflow?utm_source=chatgpt.com))

## Running the Pipeline

To execute the ZenML pipeline:

```bash
python run_pipeline.py
```

This script orchestrates the entire machine learning workflow, including data ingestion, preprocessing, model training, evaluation, and tracking using MLflow.

## Experiment Tracking with MLflow

MLflow is integrated into the pipeline to track experiments seamlessly. After running the pipeline, you can access the MLflow UI to visualize and compare experiment runs:

```bash
mlflow ui
```

By default, the MLflow UI will be accessible at `http://localhost:5000`. Here, you can explore logged parameters, metrics, artifacts, and more.

For more information on how to use MLflow with ZenML, refer to the ZenML MLflow integration guide. ([zenml.io](https://www.zenml.io/integrations/mlflow?utm_source=chatgpt.com))

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
