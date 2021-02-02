import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import EpicVideoDataset
from datasets import EpicVideoFlowDataset
from datasets import TsnDataset
from omegaconf import DictConfig
from pytorch_lightning import EvalResult
from pytorch_lightning import TrainResult
from pytorch_lightning.core.step_result import Result
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from transforms import ExtractTimeFromChannel
from transforms import GroupCenterCrop
from transforms import GroupMultiScaleCrop
from transforms import GroupNormalize
from transforms import GroupRandomHorizontalFlip
from transforms import GroupScale
from transforms import Stack
from transforms import ToTorchFormatTensor
from utils.torch_metrics import accuracy

from models.tsm import TSM
from models.tsn import MTRN
from models.tsn import TSN

TASK_CLASS_COUNTS = [("verb", 97), ("noun", 300)]
LOG = logging.getLogger(__name__)


def split_task_outputs(
    output: torch.Tensor, tasks: List[Tuple[str, int]]
) -> Dict[str, torch.Tensor]:
    offset = 0
    outputs = dict()
    for task, n_units in tasks:
        outputs[task] = output[..., offset : offset + n_units]
        offset += n_units
    return outputs


class EpicActionRecogintionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.train_gulp_dir = Path(cfg.data.train_gulp_dir)
        self.val_gulp_dir = Path(cfg.data.val_gulp_dir)
        self.test_gulp_dir = Path(cfg.data.test_gulp_dir)
        self.cfg = cfg

        channel_count = (
            3 if self.cfg.modality == "RGB" else 2 * self.cfg.data.segment_length
        )
        common_transform = Compose(
            [
                Stack(
                    bgr=self.cfg.modality == "RGB"
                    and self.cfg.data.preprocessing.get("bgr", False)
                ),
                ToTorchFormatTensor(div=self.cfg.data.preprocessing.rescale),
                GroupNormalize(
                    mean=list(self.cfg.data.preprocessing.mean),
                    std=list(self.cfg.data.preprocessing.std),
                ),
                ExtractTimeFromChannel(channel_count),
            ]
        )
        self.train_transform = Compose(
            [
                GroupMultiScaleCrop(
                    self.cfg.data.preprocessing.input_size,
                    self.cfg.data.train_augmentation.multiscale_crop_scales,
                ),
                GroupRandomHorizontalFlip(is_flow=self.cfg.modality == "Flow"),
                common_transform,
            ]
        )
        self.test_transform = Compose(
            [
                GroupScale(self.cfg.data.test_augmentation.rescale_size),
                GroupCenterCrop(self.cfg.data.preprocessing.input_size),
                common_transform,
            ]
        )

    def train_dataloader(self):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Training dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.train_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.train_transform,
            drop_problematic_metadata=True,
        )
        if self.cfg.data.get("train_on_val", False):
            LOG.info("Training on training set + validation set")
            dataset = ConcatDataset(
                [
                    dataset,
                    TsnDataset(
                        self._get_video_dataset(self.val_gulp_dir),
                        num_segments=frame_count,
                        segment_length=self.cfg.data.segment_length,
                        transform=self.train_transform,
                        drop_problematic_metadata=True,
                    ),
                ]
            )
        LOG.info(f"Training dataset size: {len(dataset)}")

        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Validation dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.val_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.test_transform,
            test_mode=True,
            drop_problematic_metadata=True,
        )
        LOG.info(f"Validation dataset size: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self):
        frame_count = self.cfg.data.get("test_frame_count", self.cfg.data.frame_count)
        LOG.info(f"Test dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.test_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.test_transform,
            test_mode=True,
            drop_problematic_metadata=True,
        )
        LOG.info(f"Test dataset size: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def _get_video_dataset(self, gulp_dir_path):
        if self.cfg.modality.lower() == "rgb":
            return EpicVideoDataset(gulp_dir_path)
        elif self.cfg.modality.lower() == "flow":
            return EpicVideoFlowDataset(gulp_dir_path)
        else:
            raise ValueError(f"Unknown modality {self.cfg.modality!r}")

    def prepare_data(self, *args, **kwargs):
        # Implemented to appease mypy
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        # Implemented to appease mypy
        pass

    def setup(self, stage: Optional[str] = None):
        # Implemented to appease mypy
        pass


class EpicActionRecognitionSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.model = load_model(cfg)
        channels = cfg.data.segment_length * (3 if cfg.modality == "RGB" else 2)
        self.example_input_array = torch.randn(  # type: ignore
            (
                1,
                cfg.data.frame_count,
                channels,
                cfg.data.preprocessing.input_size,
                cfg.data.preprocessing.input_size,
            )
        )

    def configure_optimizers(self):
        cfg = self.cfg.learning
        if cfg.optimizer.type == "SGD":
            optimizer = SGD(
                self.model.get_optim_policies(),
                lr=cfg.lr,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")
        scheduler = MultiStepLR(
            optimizer, milestones=cfg.lr_scheduler.epochs, gamma=cfg.lr_scheduler.gamma
        )
        return [optimizer], [scheduler]

    def forward(self, xs):
        return self.model(xs)

    def forward_tasks(self, xs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return split_task_outputs(self(xs), TASK_CLASS_COUNTS)

    def training_step(self, batch, batch_idx):
        step_results = self._step(batch)

        result = TrainResult(step_results["loss"])
        self.log_metrics(result, step_results, "train")
        return result

    def validation_step(self, batch, batch_idx):
        step_results = self._step(batch)
        result = EvalResult(checkpoint_on=step_results["loss"])
        self.log_metrics(result, step_results, "val")
        return result

    def test_step(self, batch, batch_idx):
        data, labels_dict = batch
        outputs = self.forward_tasks(data)

        result = EvalResult()
        filename = self.cfg.get("test.results_path", "./predictions.pt")

        result.write_dict(
            {
                "verb_output": outputs["verb"],
                "noun_output": outputs["noun"],
                "narration_id": labels_dict["narration_id"],
                "video_id": labels_dict["video_id"],
            },
            filename=filename,
        )

        return result

    def log_metrics(
        self, result: Result, step_results: Dict[str, float], step_type: str
    ) -> None:
        result.log(f"loss/{step_type}", step_results["loss"])
        for task in ["verb", "noun"]:
            result.log(f"{task}_loss/{step_type}", step_results[f"{task}_loss"])
            for k in (1, 5):
                result.log(
                    f"{task}_accuracy@{k}/{step_type}",
                    step_results[f"{task}_accuracy@{k}"],
                )

    def _step(self, batch: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        data, labels_dict = batch
        outputs: Dict[str, Tensor] = self.forward_tasks(data)
        tasks = {
            task: {
                "output": outputs[task],
                "preds": outputs[task].argmax(-1),
                "labels": labels_dict[f"{task}_class"],
                "weight": 1,
            }
            for task in ["verb", "noun"]
        }
        step_results = dict()
        loss = 0
        n_tasks = len(tasks)
        for task, d in tasks.items():
            task_loss = F.cross_entropy(d["output"], d["labels"])
            loss += d["weight"] * task_loss

            accuracy_1, accuracy_5 = accuracy(d["output"], d["labels"], ks=(1, 5))
            step_results[f"{task}_accuracy@1"] = accuracy_1
            step_results[f"{task}_accuracy@5"] = accuracy_5

            step_results[f"{task}_loss"] = task_loss
            step_results[f"{task}_preds"] = d["preds"]
            step_results[f"{task}_output"] = d["output"]
        step_results["video_ids"] = labels_dict["video_id"]
        step_results["loss"] = loss / n_tasks
        return step_results


def load_model(cfg: DictConfig) -> TSN:
    output_dim: int = sum([class_count for _, class_count in TASK_CLASS_COUNTS])
    if cfg.model.type == "TSN":
        model = TSN(
            num_class=output_dim,
            num_segments=cfg.data.frame_count,
            modality=cfg.modality,
            base_model=cfg.model.backbone,
            segment_length=cfg.data.segment_length,
            consensus_type="avg",
            dropout=cfg.model.dropout,
            partial_bn=cfg.model.partial_bn,
            pretrained=cfg.model.pretrained,
        )
    elif cfg.model.type == "MTRN":
        model = MTRN(
            num_class=output_dim,
            num_segments=cfg.data.frame_count,
            modality=cfg.modality,
            base_model=cfg.model.backbone,
            segment_length=cfg.data.segment_length,
            dropout=cfg.model.dropout,
            img_feature_dim=cfg.model.backbone_dim,
            partial_bn=cfg.model.partial_bn,
            pretrained=cfg.model.pretrained,
        )
    elif cfg.model.type == "TSM":
        model = TSM(
            num_class=output_dim,
            num_segments=cfg.data.frame_count,
            modality=cfg.modality,
            base_model=cfg.model.backbone,
            segment_length=cfg.data.segment_length,
            consensus_type="avg",
            dropout=cfg.model.dropout,
            partial_bn=cfg.model.partial_bn,
            pretrained=cfg.model.pretrained,
            shift_div=cfg.model.shift_div,
            non_local=cfg.model.non_local,
            temporal_pool=cfg.model.temporal_pool,
        )
    else:
        raise ValueError(f"Unknown model type {cfg.model.type!r}")
    if cfg.model.get("weights", None) is not None:
        if cfg.model.pretrained is not None:
            LOG.warning(
                f"model.pretrained was set to {cfg.model.pretrained!r} but "
                f"you also specified to load weights from {cfg.model.weights}."
                "The latter will take precedence."
            )

        LOG.info(f"Loading weights from {cfg.model.weights}")
        state_dict = torch.load(cfg.model.weights, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            # Person is trying to load a checkpoint with a state_dict key, so we pull
            # that out.
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    return model
