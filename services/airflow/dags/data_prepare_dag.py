from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

with DAG(dag_id="data_prepare",
         start_date=datetime(2021, 1, 1),
         catchup=False,
         schedule_interval=timedelta(minutes=5)) as dag:
    
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_data",
        external_dag_id="data_extract",
        external_task_id=None,
        mode="poke",
        timeout=600,
        poke_interval=60,
    )

    zenml_pipeline = BashOperator(
        task_id="zenml_pipeline",
        bash_command="python3 $PYTHONPATH/src/transform_data.py",
    )

    wait_for_data >> zenml_pipeline
