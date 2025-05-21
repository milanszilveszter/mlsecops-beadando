from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
import os

# Define the MLflow client
mlflow.set_tracking_uri("http://mlflow-server:5102")
client = MlflowClient()

# Set the default model name
model_name = "RandomForest"

# Path to the CSV file
dag_folder = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(dag_folder, "crop_data.csv")
print(f"CSV file path: {csv_file_path}")

def train_model():
    # Call the train endpoint with the CSV file
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('crop_data.csv', f)}
        response = requests.post("http://flask-application:8080/model/train", files=files)
    
    # Check if the request was successful
    if response.status_code != 200:
        data = response.json()
        raise Exception(f"Training failed: {data.get('error', 'Unknown error')}")
    
    # Parse response data
    data = response.json()
    new_accuracy = data["test_accuracy"]

    # Retrieve the latest staging model's test accuracy
    latest_version_info = client.get_latest_versions(model_name, stages=["Staging"])
    if latest_version_info:
        latest_version = latest_version_info[0]
        staging_accuracy = client.get_metric_history(latest_version.run_id, "test_accuracy")[-1].value

        # If new model is better or equal, transition it to Staging
        if new_accuracy >= staging_accuracy:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Archived"
            )
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
    print("Model training and evaluation completed successfully.")

# Define the Airflow DAG
with DAG(
    dag_id="daily_model_training",
    start_date=datetime(2025, 5, 19),
    schedule_interval="0 2 * * *",  # This sets the DAG to run daily at 2 AM
    catchup=False,
) as dag:

    # Define a task that calls the train_model function
    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    train_and_compare_task
