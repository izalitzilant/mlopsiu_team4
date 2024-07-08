import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from omegaconf import DictConfig

def load_data_sample():
    subprocess.run(["dvc", "push"], check=True)


with DAG(dag_id="stage_1",
         start_date=datetime(2021, 1, 1),
         catchup=False,
         schedule_interval=timedelta(minutes=5)) as dag:

    exctracted_sample = BashOperator(task_id="exctracted_sample",
                                bash_command="python3 /Users/ildarzalaliev/Desktop/mlops_rep/mlopsiu_team4/src/data.py",
                                dag=dag)

    validate_sample = BashOperator(task_id="validate_sample",
                                   bash_command="python3 /Users/ildarzalaliev/Desktop/mlops_rep/mlopsiu_team4/src/validate_data.py",
                                   dag=dag)

    version_data = BashOperator(task_id="version_data",
                                bash_command="/Users/ildarzalaliev/Desktop/mlops_rep/mlopsiu_team4/scripts/save_sample.sh ",
                                dag=dag)

    load_data = PythonOperator(task_id="load_data",
                               python_callable=load_data_sample,
                               dag=dag)

    exctracted_sample >> validate_sample >> version_data >> load_data

