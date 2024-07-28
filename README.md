![Test code workflow](https://github.com/izalitzilant/mlopsiu_team4/actions/workflows/test-code.yaml/badge.svg)
![Validate model workflow](https://github.com/izalitzilant/mlopsiu_team4/actions/workflows/validate-model.yaml/badge.svg)

# Repository structure

| File/Directory        | Description                                               |
|-----------------------|-----------------------------------------------------------|
| `README.md`           | Repo docs                                                 |
| `.gitignore`          | Gitignore file                                            |
| `requirements.txt`    | Python packages                                           |
| `configs`             | Hydra configuration management                            |
| `data`                | All data                                                  |
| `docs`                | Project docs like reports or figures                      |
| `models`              | ML models                                                 |
| `notebooks`           | Jupyter notebooks                                         |
| `outputs`             | Outputs of Hydra                                          |
| `pipelines`           | A Soft link to DAGs of Apache Airflow                     |
| `reports`             | Generated reports                                         |
| `scripts`             | Shell scripts (.sh)                                       |
| `services`            | Metadata of services (PostgreSQL, Feast, Apache Airflow, etc.) |
| `sql`                 | SQL files                                                 |
| `src`                 | Python scripts                                            |
| `tests`               | Scripts for testing Python code                           |


# How to run the pipeline

1. To set up with a project create a virtual environment with

    ```shell
    python3 -m venv .venv
    bash ./scripts/install_requirements.sh
    ```

2. Setup config by running `scripts/prepare_env.py` with your kaggle key and username:

    ```shell
    python scripts/prepare_env.py ++kaggle_key="your key" ++kaggle_username="your username"
    ```

3. Run `test_data.sh`:
   ```shell
   bash ./scripts/test_data.sh
   ```
