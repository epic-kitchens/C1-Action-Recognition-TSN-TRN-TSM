#!/usr/bin/env python
# Copyright (c) 2016, Multimedia Laboratory, The Chinese University of Hong Kong
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Notice of change:
# Modified by Will Price to support `features()` and `logits()` methods.
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torchvision
from ops.basic_ops import ConsensusModule
from ops.trn import return_TRN
from torch import nn
from torch.nn.init import constant_
from torch.nn.init import normal_

LOG = logging.getLogger(__name__)


class TSN(nn.Module):
    """
    Temporal Segment Network

    See https://arxiv.org/abs/1608.00859 for more details.

    Args:
        num_class:
            number of classes
        num_segments:
            number of frames/optical flow stacks input into the model
        modality:
            either ``rgb`` or ``flow``.
        base_model:
            backbone model architecture one of ``resnet18``, ``resnet30``,
            ``resnet50``, ``bninception``, ``inceptionv3``, ``vgg16``.
            ``bninception`` and ``resnet50`` are the most thoroughly tested.
        segment_length:
            the number of channel inputs per snippet
        consensus_type:
            the consensus function used to combined information across segments.
            one of ``avg``, ``max``, ``trn``, ``trnmultiscale``.
        dropout:
            the dropout probability. the dropout layer replaces the backbone's
            classification layer.
        img_feature_dim:
            only for trn/mtrn models. the dimensionality of the features used for
            relational reasoning.
        partial_bn:
            whether to freeze all bn layers beyond the first 2 layers.
        pretrained:
            either ``'imagenet'`` for imagenet initialised models,
            or ``'epic-kitchens'`` for weights pretrained on epic-kitchens.
    """

    def __init__(
        self,
        num_class: int,
        num_segments: int,
        modality: str,
        base_model: str = "resnet50",
        segment_length: Optional[int] = None,
        consensus_type: str = "avg",
        dropout: float = 0.7,
        img_feature_dim: int = 256,
        partial_bn: bool = True,
        pretrained: str = "imagenet",
    ):

        super(TSN, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.modality = modality
        self.arch = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.img_feature_dim = img_feature_dim
        self._enable_pbn = partial_bn
        self.pretrained = pretrained

        if segment_length is None:
            self.segment_length = 1 if modality == "RGB" else 5
        else:
            self.segment_length = segment_length

        LOG.info(
            f"""\
Initializing {self.__class__.__name__} with base model: {base_model}.

{self.__class__.__name__} Configuration:
    input_modality:     {self.modality}
    num_segments:       {self.num_segments}
    segment_length:     {self.segment_length}
    consensus_module:   {self.consensus_type}
    img_feature_dim:    {self.img_feature_dim} (only valid for TRN)
    dropout_ratio:      {self.dropout}
    partial_bn:         {partial_bn}
        """
        )

        self.base_model = self._prepare_base_model(base_model)

        self.feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name
        ).in_features
        self._prepare_tsn()

        if self.modality == "Flow":
            LOG.info("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            LOG.debug("Done. Flow model ready...")

        if consensus_type.startswith("TRN"):
            self.consensus = return_TRN(
                consensus_type, self.img_feature_dim, self.num_segments, num_class
            )
        else:
            self.consensus = ConsensusModule(consensus_type)

        if partial_bn:
            self.partialBN(True)

    def _initialise_layer(self, layer, mean=0, std=0.001):
        normal_(layer.weight, mean, std)
        constant_(layer.bias, mean)

    def _prepare_tsn(self):
        setattr(
            self.base_model,
            self.base_model.last_layer_name,
            nn.Dropout(p=self.dropout),
        )
        if self.consensus_type.startswith("TRN"):
            self.new_fc = nn.Linear(self.feature_dim, self.img_feature_dim)
        else:
            self.new_fc = nn.Linear(self.feature_dim, self.num_class)
        self._initialise_layer(self.new_fc)

    def _prepare_base_model(self, base_model_type: str) -> nn.Module:
        backbone_pretrained = "imagenet" if self.pretrained == "imagenet" else None
        if backbone_pretrained is not None:
            LOG.info(f"Loading backbone model with {backbone_pretrained} weights")
        else:
            LOG.info("Randomly initialising backbone")

        if "resnet" in base_model_type.lower():
            base_model = getattr(torchvision.models, base_model_type)(
                pretrained=backbone_pretrained
            )
            base_model.last_layer_name = "fc"
        else:
            raise ValueError("Unknown base model: {}".format(base_model_type))
        return base_model

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self) -> List[Dict[str, Any]]:
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
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
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
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
        ]

    def features(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 5:
            raise ValueError(
                f"Expected input to be 5D: (B, T, C, H, W) but was " f"{input.shape}"
            )
        # input: (B, T, C, H, W)
        batch_size, n_segments = input.shape[:2]

        # 2 channel (u, v) for Flow (but we take a stack, so it's 2 * segment_length)
        # 3 channel (r, g, b) for RGB
        channels = (3 if self.modality == "RGB" else 2) * self.segment_length

        input = input.view((-1, channels) + input.shape[-2:])
        # input: (B * T, C, H, W)
        features = self.base_model.forward(input)
        # features: (B * T, C')
        features = features.view((batch_size, n_segments) + features.shape[1:])
        # features: (B, T, C')
        return features

    def logits(self, xs: torch.Tensor) -> torch.Tensor:
        # xs: (B, T, C)
        xs = self.new_fc(xs)
        # xs: (B, T, C')
        xs = self.consensus(xs)
        # xs: (B, C')
        return xs

    def forward(
        self, xs: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # xs: (BS, T, C, H, W)
        xs = self.features(xs)
        xs = self.logits(xs)
        return xs

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
            .detach()
            .mean(dim=1, keepdim=True)
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
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model


class TRN(TSN):
    """
    Single-scale Temporal Relational Network

    See https://arxiv.org/abs/1711.08496 for more details.
    Args:
        num_class:
            Number of classes
        num_segments:
            Number of frames/optical flow stacks input into the model
        modality:
            Either ``RGB`` or ``Flow``.
        base_model:
            Backbone model architecture one of ``resnet18``, ``resnet30``,
            ``resnet50``, ``BNInception``, ``InceptionV3``, ``VGG16``.
            ``BNInception`` and ``resnet50`` are the most thoroughly tested.
        segment_length:
            The number of channel inputs per snippet
        consensus_type:
            The consensus function used to combined information across segments.
            One of ``avg``, ``max``, ``TRN``, ``TRNMultiscale``.
        dropout:
            The dropout probability. The dropout layer replaces the backbone's
            classification layer.
        img_feature_dim:
            Only for TRN/MTRN models. The dimensionality of the features used for
            relational reasoning.
        partial_bn:
            Whether to freeze all BN layers beyond the first 2 layers.
        pretrained:
            Either ``'imagenet'`` for ImageNet initialised models,
            or ``'epic-kitchens'`` for weights pretrained on EPIC-Kitchens.
    """

    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model="resnet50",
        segment_length=None,
        dropout=0.7,
        img_feature_dim=256,
        partial_bn=True,
        pretrained="imagenet",
    ):

        super().__init__(
            num_class=num_class,
            num_segments=num_segments,
            modality=modality,
            base_model=base_model,
            segment_length=segment_length,
            consensus_type="TRN",
            dropout=dropout,
            img_feature_dim=img_feature_dim,
            partial_bn=partial_bn,
            pretrained=pretrained,
        )


class MTRN(TSN):
    """
    Multi-scale Temporal Relational Network

    See https://arxiv.org/abs/1711.08496 for more details.
    Args:
        num_class:
            Number of classes
        num_segments:
            Number of frames/optical flow stacks input into the model
        modality:
            Either ``RGB`` or ``Flow``.
        base_model:
            Backbone model architecture one of ``resnet18``, ``resnet30``,
            ``resnet50``, ``BNInception``, ``InceptionV3``, ``VGG16``.
            ``BNInception`` and ``resnet50`` are the most thoroughly tested.
        segment_length:
            The number of channel inputs per snippet
        consensus_type:
            The consensus function used to combined information across segments.
            One of ``avg``, ``max``, ``TRN``, ``TRNMultiscale``.
        dropout:
            The dropout probability. The dropout layer replaces the backbone's
            classification layer.
        img_feature_dim:
            Only for TRN/MTRN models. The dimensionality of the features used for
            relational reasoning.
        partial_bn:
            Whether to freeze all BN layers beyond the first 2 layers.
        pretrained:
            Either ``'imagenet'`` for ImageNet initialised models,
            or ``'epic-kitchens'`` for weights pretrained on EPIC-Kitchens.
    """

    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model="resnet50",
        segment_length=None,
        dropout=0.7,
        img_feature_dim=256,
        partial_bn=True,
        pretrained="imagenet",
    ):

        super().__init__(
            num_class=num_class,
            num_segments=num_segments,
            modality=modality,
            base_model=base_model,
            segment_length=segment_length,
            consensus_type="TRNMultiscale",
            dropout=dropout,
            img_feature_dim=img_feature_dim,
            partial_bn=partial_bn,
            pretrained=pretrained,
        )
