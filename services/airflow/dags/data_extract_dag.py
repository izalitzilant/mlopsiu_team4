import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from omegaconf import DictConfig


with DAG(dag_id="data_extract",
         start_date=datetime(2021, 1, 1),
         catchup=False,
         schedule_interval=timedelta(minutes=5)) as dag:

    exctracted_sample = BashOperator(task_id="exctracted_sample",
                                bash_command="python3 $PYTHONPATH/data.py",
                                dag=dag)

    validate_sample = BashOperator(task_id="validate_sample",
                                   bash_command="python3 $PYTHONPATH/validate_data.py",
                                   dag=dag)

    version_data = BashOperator(task_id="version_data",
                                bash_command="python3 $PYTHONPATH/version_data.py",
                                dag=dag)

    exctracted_sample >> validate_sample >> version_data

