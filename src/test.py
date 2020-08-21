import argparse
import logging
import os
from pathlib import Path

import colorlog
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from systems import EpicActionRecogintionDataModule
from systems import EpicActionRecognitionSystem

parser = argparse.ArgumentParser(
    description="Test model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("results", type=Path)
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument(
    "--n-frames",
    type=int,
    help="Overwrite number of frames to feed model, defaults to the "
    "data.test_frame_count or data.frame_count if the former is not present",
)
parser.add_argument(
    "--batch-size",
    type=int,
    help="Overwrite the batch size for loading data, defaults to learning.batch_size",
)
parser.add_argument(
    "--datadir",
    default=None,
    help="Overwrite data directory in checkpoint. Useful when testing a checkpoint "
    "trained on a different machine.",
)

LOG = logging.getLogger("test")


def main(args):
    logging.basicConfig(level=logging.INFO)

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
    )

    logger = colorlog.getLogger("example")
    logger.addHandler(handler)

    ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    # Publicly released checkpoints use dicts for longevity, so we need to wrap them
    # up in an OmegaConf object as this is what EpicActionRecognitionSystem expects.
    cfg = OmegaConf.create(ckpt["hyper_parameters"])
    OmegaConf.set_struct(cfg, False)  # allow writing arbitrary keys without raising
    # exceptions
    cfg.data._root_gulp_dir = os.getcwd()  # set default root gulp dir to prevent
    # exceptions on instantiating the EpicActionRecognitionSystem

    system = EpicActionRecognitionSystem(cfg)
    system.load_state_dict(ckpt["state_dict"])
    if not cfg.get("log_graph", True):
        system.example_input_array = None

    if args.n_frames is not None:
        cfg.data.test_frame_count = args.n_frames
    if args.batch_size is not None:
        cfg.learning.batch_size = args.batch_size
    if args.datadir is not None:
        data_dir_key = f"{args.split}_gulp_dir"
        cfg.data[data_dir_key] = args.datadir

    # Since pytorch-lightning can't support writing results when using DP or DDP
    LOG.info("Disabling distributed backend")
    cfg.trainer.distributed_backend = None

    n_gpus = 1
    LOG.info(f"Overwriting number of GPUs to {n_gpus}")
    cfg.trainer.gpus = n_gpus
    cfg["test.results_path"] = str(args.results)

    data_module = EpicActionRecogintionDataModule(cfg)
    if args.split == "val":
        dataloader = data_module.val_dataloader()
    elif args.split == "test":
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(
            f"Split {args.split!r} is not a recognised dataset split to " f"test on."
        )
    trainer = Trainer(**cfg.trainer)
    trainer.test(system, test_dataloaders=dataloader)


if __name__ == "__main__":
    main(parser.parse_args())
