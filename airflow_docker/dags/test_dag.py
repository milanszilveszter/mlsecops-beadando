from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests

# Function to call the API
def call_api_function():
    """
    This function makes a request to an API and returns the response
    """
    api_url = "http://flask-application:8080/hello"
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.text
        print(f"API call successful. Data: {data}")
        return data
    else:
        print(f"API call failed with status code: {response.status_code}")
        return None

# Create DAG
dag = DAG(
    'api_call_dag',
    description='DAG to call an API',
    schedule=timedelta(minutes=30),
    start_date=datetime(2025, 5, 20),
    catchup=False,
    tags=['api', 'example'],
)

# Create task
call_api_task = PythonOperator(
    task_id='call_api',
    python_callable=call_api_function,
    provide_context=True,
    dag=dag,
)
