import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def app(cfg: DictConfig):
    """
    Entry point of application
    """
    pass
    # print(cfg.db.user)
    # print(cfg.db.port)
    print(OmegaConf.to_yaml(cfg))
    # print(cfg)

if __name__ == "__main__":
    app()
