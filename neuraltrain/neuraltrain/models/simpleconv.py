# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleConv, taken and slightly modified from brainmagick.

TODO:
- Move unit tests as well
- Simplify PositionGetter to work without MNE object?
- Simplify ConvSequence
- Test out temporal aggregation version
"""

import logging
import typing as tp
from functools import partial

import torch
from torch import nn
from torchvision.ops import MLP

from .base import BaseModelConfig
from .common import (
    BahdanauAttention,
    ChannelMergerConfig,
    FourierEmbConfig,
    LayerScale,
    MlpConfig,
    SubjectLayersConfig,
)
from .transformer import TransformerEncoderConfig

logger = logging.getLogger(__name__)


class ConvSequence(nn.Module):
    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: int | None = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0.0,
        skip: bool = False,
        scale: float | None = None,
        rewrite: bool = False,
        activation_on_last: bool = True,
        post_skip: bool = False,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: tp.Any = None,
    ) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0.0 < dropout_input < 1.0
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(
                Conv(
                    chin,
                    chout,
                    kernel,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=groups if k > 0 else 1,
                )
            )
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context),
                        act,
                    )
                )
            else:
                self.glus.append(None)  # type: ignore

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class SimpleConvConfig(BaseModelConfig):
    name: tp.Literal["SimpleConv"] = "SimpleConv"

    # Channels
    hidden: int = 16
    # Overall structure
    depth: int = 4
    linear_out: bool = False
    complex_out: bool = False
    # Conv layer
    kernel_size: int = 5
    growth: float = 1.0
    dilation_growth: int = 2
    dilation_period: int | None = None
    skip: bool = False
    post_skip: bool = False
    scale: float | None = None
    rewrite: bool = False
    groups: int = 1
    glu: int = 0
    glu_context: int = 0
    glu_glu: bool = True
    gelu: bool = False
    # Dropouts, BN, activations
    dropout: float = 0.0
    dropout_rescale: bool = True
    conv_dropout: float = 0.0
    dropout_input: float = 0.0
    batch_norm: bool = False
    relu_leakiness: float = 0.0
    # Optional transformer
    transformer_config: TransformerEncoderConfig | None = None
    # Subject-specific settings
    subject_layers_config: SubjectLayersConfig | None = None
    subject_layers_dim: tp.Literal["input", "hidden"] = "hidden"
    # Channel attention for multi-montage support
    merger_config: ChannelMergerConfig | None = ChannelMergerConfig(
        n_virtual_channels=270,
        fourier_emb_config=FourierEmbConfig(
            n_freqs=None,
            total_dim=2048,
            n_dims=2,
        ),
        dropout=0.2,
        usage_penalty=0.0,
        per_subject=False,
        embed_ref=False,
    )
    # Architectural details
    initial_linear: int = 0
    initial_depth: int = 1
    initial_nonlin: bool = False
    backbone_out_channels: int | None = None  # If provided, the output of the
    # backbone (i.e. layer before the output heads) will have this dimensionality

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return SimpleConv(n_in_channels, n_outputs, config=self)


class SimpleConv(nn.Module):
    def __init__(
        self,
        # Channels
        in_channels: int,
        out_channels: int,
        config: SimpleConvConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else SimpleConvConfig()

        self.out_channels = out_channels
        self.backbone_out_channels = (
            out_channels
            if config.backbone_out_channels is None
            else config.backbone_out_channels
        )

        self.linear = nn.Linear(config.hidden, self.backbone_out_channels)

        activation: nn.Module | tp.Callable
        if config.gelu:
            activation = nn.GELU
        elif config.relu_leakiness:
            activation = partial(nn.LeakyReLU, config.relu_leakiness)
        else:
            activation = nn.ReLU

        assert config.kernel_size % 2 == 1, "For padding to work, this must be verified"

        self.dropout = None

        self.initial_linear = None
        if config.dropout > 0.0:
            raise NotImplementedError("To be reimplemented here.")
            # self.dropout = ChannelDropout(dropout, dropout_rescale)

        self.merger = None
        if config.merger_config is not None:
            self.merger = config.merger_config.build()
            in_channels = config.merger_config.n_virtual_channels

        if config.initial_linear:
            init: list[nn.Module | tp.Callable] = [
                nn.Conv1d(in_channels, config.initial_linear, 1)
            ]
            for _ in range(config.initial_depth - 1):
                init += [
                    activation(),
                    nn.Conv1d(config.initial_linear, config.initial_linear, 1),
                ]
            if config.initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)  # type: ignore[arg-type]
            in_channels = config.initial_linear

        self.subject_layers = None
        if config.subject_layers_config:
            dim = {"hidden": config.hidden, "input": in_channels}[
                config.subject_layers_dim
            ]
            self.subject_layers = config.subject_layers_config.build(in_channels, dim)
            in_channels = dim

        # compute the sequences of channel sizes
        sizes = [in_channels]
        sizes += [
            int(round(config.hidden * config.growth**k)) for k in range(config.depth)
        ]

        params: tp.Dict[str, tp.Any]
        params = dict(
            kernel=config.kernel_size,
            stride=1,
            leakiness=config.relu_leakiness,
            dropout=config.conv_dropout,
            dropout_input=config.dropout_input,
            batch_norm=config.batch_norm,
            dilation_growth=config.dilation_growth,
            groups=config.groups,
            dilation_period=config.dilation_period,
            skip=config.skip,
            post_skip=config.post_skip,
            scale=config.scale,
            rewrite=config.rewrite,
            glu=config.glu,
            glu_context=config.glu_context,
            glu_glu=config.glu_glu,
            activation=activation,
        )

        final_channels = sizes[-1]

        self.final: nn.Module | nn.Sequential | None = None
        pad = 0
        kernel = 1
        stride = 1

        if config.linear_out:
            assert not config.complex_out
            self.final = nn.ConvTranspose1d(
                final_channels, self.backbone_out_channels, kernel, stride, pad
            )
        elif config.complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(
                    2 * final_channels, self.backbone_out_channels, kernel, stride, pad
                ),
            )
        else:
            params["activation_on_last"] = False
            sizes[-1] = self.backbone_out_channels

        if config.transformer_config:
            sizes[-1] = config.hidden

        self.encoder = ConvSequence(sizes, **params)

        self.transformer = None
        if config.transformer_config:
            self.transformer = config.transformer_config.build(dim=config.hidden)

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = x.shape[-1]

        # if self.dropout is not None:
        #     x = self.dropout(x, batch)

        if self.merger is not None:
            x = self.merger(x, subject_ids, channel_positions)

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)

        x = self.encoder(x)

        if self.final is not None:
            x = self.final(x)
        assert x.shape[-1] >= length
        x = x[:, :, :length]

        if self.transformer:
            x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
            x = x.permute(0, 2, 1)
            x = self.linear(x)
            x = x.permute(0, 2, 1)

        return x


class SimpleConvTimeAggConfig(SimpleConvConfig):
    name: tp.Literal["SimpleConvTimeAgg"] = "SimpleConvTimeAgg"  # type: ignore

    # Temporal aggregation
    time_agg_out: tp.Literal["gap", "linear", "att"] = "gap"
    # Output head(s)
    output_head_config: MlpConfig | dict[str, MlpConfig] | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return SimpleConvTimeAgg(n_in_channels, n_outputs, config=self)


class SimpleConvTimeAgg(SimpleConv):
    """SimpleConv with temporal aggregation layer and potentially two output heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SimpleConvTimeAggConfig | None = None,
    ):
        config = config if config is not None else SimpleConvTimeAggConfig()
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, config=config
        )

        # Output aggregation layer
        self.time_agg_out: nn.Module | None
        if config.time_agg_out == "gap":
            self.time_agg_out = nn.AdaptiveAvgPool1d(1)
        elif config.time_agg_out == "linear":
            self.time_agg_out = nn.LazyLinear(1)
        elif config.time_agg_out == "att":
            self.time_agg_out = BahdanauAttention(input_size=None, hidden_size=256)
        else:
            self.time_agg_out = None

        # Separate output head(s)
        self.output_head: None | MLP | dict[str, MLP]
        if config.output_head_config is None:
            self.output_head = None
        else:
            if self.time_agg_out is None:
                raise NotImplementedError("Output heads require temporal aggregation.")
            if isinstance(config.output_head_config, MlpConfig):
                self.output_head = config.output_head_config.build(
                    input_size=self.backbone_out_channels
                )
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(
                        input_size=self.backbone_out_channels
                    )

    def forward(  # type: ignore
        self, x, subject_ids=None, channel_positions=None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = super().forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )

        if self.time_agg_out is not None:
            x = self.time_agg_out(x)
            if x.ndim == 3:
                x = x.squeeze(2)  # Remove singleton dimension

        # Apply output heads (e.g. for separate CLIP and MSE losses)
        if isinstance(self.output_head, MLP):
            x = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            x = {name: head(x) for name, head in self.output_head.items()}

        return x
