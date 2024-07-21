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

2. Follow [README](./configs/README.md) in ./configs and setup the kaggle.yaml

3. Run `test_data.sh`:
   ```shell
   bash ./scripts/test_data.sh
   ```
