# Repository structure

├───README.md          # Repo docs <br> 
├───.gitignore         # gitignore file <br> 
├───requirements.txt   # Python packages <br>
├───configs            # Hydra configuration management <br>
├───data               # All data <br>
├───docs               # Project docs like reports or figures <br>
├───models             # ML models <br>
├───notebooks          # Jupyter notebooks <br>
├───outputs            # Outputs of Hydra <br>
├───pipelines          # A Soft link to DAGs of Apache Airflow <br>
├───reports            # Generated reports <br> 
├───scripts            # Shell scripts (.sh) <br>
├───services           # Metadata of services (PostgreSQL, Feast, Apache airflow, ...etc) <br>
├───sql                # SQL files <br>
├───src                # Python scripts <br>
└───tests              # Scripts for testing Python code <br>

# How to run the pipeline

1. To set up with a project create a virtual environment with

    ```shell
    python3 -m venv .venv
    bash ./scripts/install_requirements.sh
    ```

2. Follow [README](./configs/README.md) in ./configs and setup the kaggle.yaml

3. Run `test_data.sh`:
   ```shell
   bash ./scripts/test_data.sh
   ```
