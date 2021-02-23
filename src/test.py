from collections import defaultdict

import argparse
import logging
import os
import pickle
from pathlib import Path

import colorlog
import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Callback, Trainer
from typing import Any, Dict, List, Sequence, Union

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


class ResultsSaver(Callback):
    def __init__(self):
        super().__init__()
        self.results: Dict[str, Dict[str, List[Any]]] = dict()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._store_batch_results("test", outputs)

    def _store_batch_results(
        self, dataset_name: str, batch_outputs: Dict[str, Sequence[Any]]
    ):
        if dataset_name not in self.results:
            self.results[dataset_name] = {k: [] for k in batch_outputs.keys()}

        for k, vs in batch_outputs.items():
            if isinstance(vs, torch.Tensor):
                vs = vs.detach().cpu().numpy()
            self.results[dataset_name][k].extend(vs)

    def save_results(self, dataset_name: str, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results_dict = self.results[dataset_name]
        new_results_dict = {
            k: np.stack(vs)
            for k, vs in results_dict.items()
        }

        with open(filepath, "wb") as f:
            pickle.dump(new_results_dict, f)


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
        # MTRN can't be traced due to the model stochasticity so causes a JIT tracer
        # error, we allow you to prevent the tracer from running to log the graph when
        # the summary writer is created
        try:
            delattr(system, "example_input_array")
        except AttributeError:
            pass

    if args.n_frames is not None:
        cfg.data.test_frame_count = args.n_frames
    if args.batch_size is not None:
        cfg.learning.batch_size = args.batch_size
    if args.datadir is not None:
        data_dir_key = f"{args.split}_gulp_dir"
        cfg.data[data_dir_key] = args.datadir

    # Since we don't support writing results when using DP or DDP
    LOG.info("Disabling DP/DDP")
    cfg.trainer.accelerator = None

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

    saver = ResultsSaver()
    trainer = Trainer(**cfg.trainer, callbacks=[saver])
    trainer.test(system, test_dataloaders=dataloader)
    saver.save_results("test", args.results)


if __name__ == "__main__":
    main(parser.parse_args())
