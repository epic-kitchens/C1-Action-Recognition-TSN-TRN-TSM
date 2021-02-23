import logging

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from systems import EpicActionRecogintionDataModule

LOG = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="tsn_rgb")
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    data_module = EpicActionRecogintionDataModule(cfg)
    loader = data_module.train_dataloader()
    for batch in tqdm(loader):
        pass


if __name__ == "__main__":
    main()
