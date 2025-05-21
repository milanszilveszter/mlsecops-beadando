# MLSecOps beadandó (2024/25/2)
**Készítette:** Szilveszter Milán <br>

## Könyvtárak és fileok
1. <code>train_pipeline.ipynb</code>
    - Tantási folyamat notebookja (EDA + model train).
2. <code>docker-compose.yaml</code>
    - Docker compose, ami a servicek dockerfájljaiból előállítja a konténereket.    
3. <code>/flask_docker</code>
    - Flask REST szerverhez szükséges kódok + dockerfile.
4. <code>/mlflow_docker</code>
    - MLflow szerverhez szükséges könyvtárak (lokális tanítások, artifactok) + dockerfile.   
5. <code>/airflow_docker</code>
    - Airflow-hoz szükséges DAG-ok + dockerfile.
6. <code>/streamlit_evidentlyai</code>
    - Streamlit dashboardhoz adatok, illetve Evidently AI notebook és kimentett riportok. 
7. <code>/data</code>
    - Tanításhoz használt adat(ok).
8. <code>/artifacts</code>
    - Tanítás során kimentett artifact-ok.

## Docker compose
<code>docker compose up --build -d</code>

## Elérési útvonalak
- **Flask:** <code>https://localhost:8080</code><br>
- **MLflow:** <code>https://localhost:5102</code><br>
- **Airflow:** <code>https://localhost:8090</code><br>
    - **User:** admin
    - **Password:** Admin1234

## További parancsok
### Streamlit dashboard:
<code>cd streamlit_evidentlyai</code><br>
<code>streamlit run monitor_with_streamlit_train_data.py</code>