# sample 50k records from the dataset
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="datasets")
def main(cfg: DictConfig) -> None:
    print('Sampling 50k records from the dataset... ', end='')
    df = pd.read_csv(cfg.dataset.train_dataset_path)
    df = df.sample(50000)
    sampled_df_filename = 'train50K'
    df.to_csv(cfg.dataset.sampled_dataset_path + 'train50K.csv', index=False)
    print('Done!')

if __name__ == "__main__":
    main()
