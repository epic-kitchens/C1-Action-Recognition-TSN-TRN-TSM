import logging

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from systems import EpicActionRecogintionDataModule
from systems import EpicActionRecognitionSystem

LOG = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="tsn_rgb")
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    system = EpicActionRecognitionSystem(cfg)
    if not cfg.get("log_graph", True):
        # MTRN can't be traced due to the model stochasticity so causes a JIT tracer
        # error, we allow you to prevent the tracer from running to log the graph when
        # the summary writer is created
        try:
            delattr(system, 'example_input_array')
        except AttributeError:
            pass
    data_module = EpicActionRecogintionDataModule(cfg)
    checkpoint_callback = ModelCheckpoint(save_top_k=None, monitor=None)
    # with ipdb.launch_ipdb_on_exception():
    trainer = Trainer(
        callbacks=[], checkpoint_callback=checkpoint_callback, **cfg.trainer
    )
    trainer.fit(system, datamodule=data_module)


if __name__ == "__main__":
    main()
