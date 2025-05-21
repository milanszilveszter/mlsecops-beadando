FROM apache/airflow:2.10.5-python3.11

USER root
COPY dags /opt/airflow/dags
COPY requirements.txt /opt/airflow/requirements.txt

USER airflow
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt