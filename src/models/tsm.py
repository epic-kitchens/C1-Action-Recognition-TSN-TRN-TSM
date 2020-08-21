# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import logging
import re

import numpy as np
import torch
import torchvision
from ops.basic_ops import ConsensusModule
from ops.temporal_shift import make_temporal_shift
from torch import nn
from torch.nn.init import constant_
from torch.nn.init import normal_
from torch.utils import model_zoo

LOG = logging.getLogger(__name__)


def strip_module_prefix(state_dict):
    return {re.sub("^module.", "", k): v for k, v in state_dict.items()}


class TSM(nn.Module):
    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model="resnet101",
        segment_length=None,
        consensus_type="avg",
        before_softmax=True,
        dropout=0.8,
        crop_num=1,
        partial_bn=True,
        pretrained="imagenet",
        is_shift=True,
        shift_div=8,
        shift_place="blockres",
        fc_lr5=False,
        temporal_pool=False,
        non_local=False,
    ):
        super().__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.pretrained = pretrained

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if segment_length is None:
            self.segment_length = 1 if modality == "RGB" else 5
        else:
            self.segment_length = segment_length
        LOG.info(
            f"""
    Initializing {self.__class__.__name__} with base model: {base_model}.

    {self.__class__.__name__} Configuration:
        input_modality:     {self.modality}
        num_segments:       {self.num_segments}
        segment_length:     {self.segment_length}
        consensus_module:   {self.consensus_type}
        dropout_ratio:      {self.dropout}
            """
        )

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        if self.modality == "Flow":
            LOG.info("Converting model to take operate on optical flow")
            self.base_model = self._construct_flow_model(self.base_model)

        self.consensus = ConsensusModule(consensus_type)

        if self.pretrained == "kinetics":
            LOG.info("Loading kinetics pretrained weights")
            if self.modality.lower() == "rgb":
                sd = strip_module_prefix(
                    model_zoo.load_url(
                        "https://file.lzhu.me/projects/tsm/models/"
                        "TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth"
                    )["state_dict"]
                )
                del sd["new_fc.weight"]
                del sd["new_fc.bias"]
                missing, unexpected = self.load_state_dict(sd, strict=False)
                if len(missing) > 0:
                    LOG.warning(f"Missing keys in checkpoint: {missing}")
                if len(unexpected) > 0:
                    LOG.warning(f"Unexpected keys in checkpoint: {unexpected}")
                LOG.info("Loading kinetics pretrained RGB weights")
            elif self.modality.lower() == "flow":
                sd = strip_module_prefix(
                    model_zoo.load_url(
                        "https://file.lzhu.me/projects/tsm/models/"
                        "TSM_kinetics_Flow_resnet50_shift8_blockres_avg_segment8_e50.pth"
                    )["state_dict"]
                )
                del sd["new_fc.weight"]
                del sd["new_fc.bias"]
                missing, unexpected = self.load_state_dict(sd, strict=False)
                if len(missing) > 0:
                    LOG.warning(f"Missing keys in checkpoint: {missing}")
                if len(unexpected) > 0:
                    LOG.warning(f"Unexpected keys in checkpoint: {unexpected}")
                LOG.info("Loading kinetics pretrained flow weights")
            else:
                raise ValueError(f"Unknown modality {self.modality}")

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name
        ).in_features
        if self.dropout == 0:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Linear(feature_dim, num_class),
            )
            self.new_fc = None
        else:
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(p=self.dropout),
            )
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std
            )
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, "weight"):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        LOG.info(f"base model: {base_model}")
        backbone_pretrained = "imagenet" if self.pretrained == "imagenet" else None
        if self.pretrained and backbone_pretrained is not None:
            LOG.info(f"Loading backbone model with {backbone_pretrained} weights")
        elif self.pretrained is None and backbone_pretrained is None:
            LOG.info("Randomly initialising backbone")

        if "resnet" in base_model:
            self.base_model = getattr(torchvision.models, base_model)(
                pretrained=backbone_pretrained
            )
            if self.is_shift:
                LOG.info("Adding temporal shift...")

                make_temporal_shift(
                    self.base_model,
                    self.num_segments,
                    n_div=self.shift_div,
                    place=self.shift_place,
                    temporal_pool=self.temporal_pool,
                )

            if self.non_local:
                LOG.info("Adding non-local module...")
                from ..ops.non_local import make_non_local

                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = "fc"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == "Flow":
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
        else:
            raise ValueError(f"Unknown base model: {base_model!r}")

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSM, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            LOG.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if (
                isinstance(m, torch.nn.Conv2d)
                or isinstance(m, torch.nn.Conv1d)
                or isinstance(m, torch.nn.Conv3d)
            ):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".format(
                            type(m)
                        )
                    )

        return [
            {
                "params": first_conv_weight,
                "lr_mult": 5 if self.modality == "Flow" else 1,
                "decay_mult": 1,
                "name": "first_conv_weight",
            },
            {
                "params": first_conv_bias,
                "lr_mult": 10 if self.modality == "Flow" else 2,
                "decay_mult": 0,
                "name": "first_conv_bias",
            },
            {
                "params": normal_weight,
                "lr_mult": 1,
                "decay_mult": 1,
                "name": "normal_weight",
            },
            {
                "params": normal_bias,
                "lr_mult": 2,
                "decay_mult": 0,
                "name": "normal_bias",
            },
            {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "BN scale/shift"},
            {"params": custom_ops, "lr_mult": 1, "decay_mult": 1, "name": "custom_ops"},
            # for fc
            {"params": lr5_weight, "lr_mult": 5, "decay_mult": 1, "name": "lr5_weight"},
            {"params": lr10_bias, "lr_mult": 10, "decay_mult": 0, "name": "lr10_bias"},
        ]

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.segment_length

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view(
                    (-1, self.num_segments // 2) + base_out.size()[1:]
                )
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(
            filter(
                lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
            )
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.segment_length,) + kernel_size[2:]
        new_kernels = (
            params[0]
            .data.mean(dim=1, keepdim=True)
            .expand(new_kernel_size)
            .contiguous()
        )

        new_conv = nn.Conv2d(
            2 * self.segment_length,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return base_model
