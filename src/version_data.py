import subprocess
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def version_data(cfg: DictConfig):
    data_sample_path = os.path.join(cfg.paths.root_path, 'data', 'samples', cfg.datasets.sample_filename)
    config_path = os.path.join(cfg.paths.root_path, 'configs', 'datasets.yaml')
    root_path = cfg.paths.root_path

    subprocess.run(["dvc", "add", data_sample_path], cwd=root_path)
    subprocess.run(["git", "add", f"{data_sample_path}.dvc"], cwd=root_path)
    subprocess.run(["git", "add", config_path], cwd=root_path)

    subprocess.run(["git", "commit", "-m", cfg.datasets.message], cwd=root_path)
    subprocess.run(["git", "push"], cwd=root_path)

    subprocess.run(["git", "tag", "-a", cfg.datasets.version, "-m", cfg.datasets.message], cwd=root_path)
    subprocess.run(["git", "push", "--tags"], cwd=root_path)

    subprocess.run(["dvc", "push"], cwd=root_path)

    print(f"Data version {cfg.datasets.version} committed and tagged successfully!")

if __name__ == "__main__":
    version_data()
