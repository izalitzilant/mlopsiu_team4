from datetime import datetime, timedeltaeho

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from src.data import sample_data
from src.validate_data import validate_initial_dataset


with DAG(dag_id="stage_1",
         start_date=datetime(2021, 1, 1),
         schedule_interval="* * * * *") as dag:

    exctracted_sample = PythonOperator(task_id="exctracted_sample",
                                       python_callable=sample_data,
                                       dag=dag)

    validate_sample = PythonOperator(task_id="validate_sample",
                                     python_callable=validate_initial_dataset,
                                     dag=dag)

    version_data = BashOperator(task_id="version_data",
                                bash_command="../../scripts/save_sample.sh",
                                dag=dag)

if __name__ == '__main__':
    dag.test()
