# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Common modules to be used with brain models."""

import math
import typing as tp
from collections import deque

import torch
from torch import nn
from torchvision.ops import MLP

from neuralset.features.neuro import ChannelPositions

from .base import BaseModelConfig


class BahdanauAttention(nn.Module):
    """Bahdanau attention from [1]_.
    Implementation inspired from pytorch's seq2seq tutorial:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-decoder
    .. [1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by
           jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        if input_size is None:
            self.Wa = nn.LazyLinear(hidden_size)
            self.Ua = nn.LazyLinear(hidden_size)
        else:
            self.Wa = nn.Linear(input_size, hidden_size)
            self.Ua = nn.Linear(input_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, keys, queries=None):
        """
        Parameters
        ----------
        query :
            Query tensor of shape (batch_size, n_features, n_times).
        """
        keys = keys.transpose(2, 1)  # (B, F, T) -> (B, T, F)
        sum_ = self.Wa(keys)
        if queries is not None:
            queries = queries.transpose(2, 1)
            assert queries.shape == keys.shape
            sum_ += self.Ua(queries)

        scores = self.Va(torch.tanh(sum_))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        context = context.transpose(2, 1)  # (B, 1, F) -> (B, F, 1)

        return context


class SubjectLayersConfig(BaseModelConfig):
    name: tp.Literal["SubjectLayers"] = "SubjectLayers"
    n_subjects: int = 200
    init_id: bool = False
    mode: tp.Literal["gather", "for_loop"] = "gather"

    def build(self, in_channels: int, out_channels: int) -> nn.Module:
        kwargs = self.model_dump()
        del kwargs["name"]
        return SubjectLayers(in_channels, out_channels, **kwargs)


class SubjectLayers(nn.Module):
    """Per subject linear projection.

    Parameters
    ----------
    in_channels :
        Number of input channels.
    out_channels :
        Number of output channels.
    n_subjects :
        Number of subjects to initialize weights for.
    init_id :
        If True, initialize the projection matrices with the identity.
    mode :
        How to apply the linear projection. With "gather" (original implementation), a tensor of
        shape (batch_size, in_channels, out_channels) containing the projection matrices for each
        example in the batch is first created. This tensor can be very large when the number of
        channels is high (e.g. when using on fMRI data with many input voxels). In this case, it
        may be better to use "for_loop": this will loop over each unique subject in the batch to
        apply the projection separately.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int = 200,
        init_id: bool = False,
        mode: tp.Literal["gather", "for_loop"] = "gather",
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        if init_id:
            if in_channels != out_channels:
                raise ValueError(
                    "in_channels and out_channels must be the same for identity initialization."
                )
            self.weights.data[:] = torch.eye(in_channels)[None]
        else:
            self.weights.data.normal_()
        self.weights.data *= 1 / in_channels**0.5
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, in_channels, n_times)
        subjects: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:  # (batch_size, out_channels, n_times)
        N, C, D = self.weights.shape
        assert subjects.max() < N, (
            "Subject index higher than number of subjects used to initialize the weights."
        )

        if self.mode == "gather":
            weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
            out = torch.einsum("bct,bcd->bdt", x, weights)
        elif self.mode == "for_loop":
            B, _, T = x.shape
            out = torch.empty((B, D, T), device=x.device)
            for subject in subjects.unique():
                mask = subjects.reshape(-1) == subject
                out[mask] = torch.einsum("bct,cd->bdt", x[mask], self.weights[subject])
        else:
            raise NotImplementedError()

        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class FourierEmbConfig(BaseModelConfig):
    """Configuration for Fourier positional embedding.

    Parameters
    ----------
    n_freqs :
        Number of frequencies (harmonics) used to encode **one** dimension.
    total_dim :
        If provided instead of `n_freqs`, this will be used to compute the number of
        frequencies following this relationship:

            n_freqs = (total_dim / 2) ** (1 / n_dims)

        If the resulting `n_freqs` is not an integer an exception will be raised.
    n_dims :
        Number of dimensions to embed. This should be 2 for 2D positions (e.g. MNE layouts) or 3
        for 3D positions (e.g. MNE montages).
    margin :
        How much to extend the range of the embedding to avoid edge effects.
    """

    name: tp.Literal["FourierEmb"] = "FourierEmb"

    n_freqs: int | None = 12
    total_dim: int | None = None
    n_dims: int = 2
    margin: float = 0.2

    def build(self) -> nn.Module:
        if self.total_dim is not None and self.n_freqs is None:
            n_freqs = (self.total_dim / 2) ** (1 / self.n_dims)
            if abs(n_freqs - round(n_freqs)) > 1e-6:  # Check if n_freqs is integer
                raise ValueError("(total_dim / 2) ** (1 / n_dims) must be an integer.")
            n_freqs = round(n_freqs)
        elif self.n_freqs is not None and self.total_dim is None:
            n_freqs = self.n_freqs
        else:
            raise ValueError("Exactly one of n_freqs and total_dim must be provided.")

        return FourierEmb(
            n_freqs=n_freqs,
            n_dims=self.n_dims,
            margin=self.margin,
        )


class FourierEmb(nn.Module):
    """Fourier positional embedding.

    Unlike traditional embedding this is not using exponential periods for cosines and sinuses, but
    typical `2 pi k` which can represent any function over [0, 1]. As this function would be
    necessarily periodic, we take a bit of margin and do over e.g. [-0.2, 1.2].
    """

    def __init__(
        self,
        n_freqs: int,
        n_dims: int,
        margin: float,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_dims = n_dims
        self.margin = margin

        # Precompute sin/cos arguments
        freqs = torch.arange(n_freqs)
        width = 1 + 2 * self.margin
        pos = 2 * math.pi * freqs / width
        self.register_buffer("pos", pos)

    @property
    def total_dim(self) -> int:
        """Total dimension of the embedding."""
        return (self.n_freqs**self.n_dims) * 2

    @staticmethod
    def _outer_sum(x: torch.Tensor) -> torch.Tensor:
        """Outer sum between the last dimensions of `x`.

        x.shape[-1] is expected to match the dimensions of the grid, between which the outer sum
        is computed. For example, if x.shape[-1] == 2, the outer sum will be computed between the
        last two dimensions of `x`, i.e. x[..., 0] and x[..., 1].
        """
        inds = deque([slice(None)] + [None] * (x.shape[-1] - 1))
        out = x[..., 0][..., *inds]
        for i in range(1, x.shape[-1]):
            inds.rotate()
            out = out + x[..., i][..., *inds]
        return out

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        *O, D = positions.shape
        assert D == self.n_dims, f"Expected {self.n_dims} positions, but got {D}."
        positions = positions + self.margin
        locs = torch.einsum("bcd,f->bcfd", positions, self.pos)
        loc_grid = self._outer_sum(locs).view(*O, -1)
        emb = torch.cat(
            [
                torch.cos(loc_grid),
                torch.sin(loc_grid),
            ],
            dim=-1,
        )
        return emb


class ChannelMergerConfig(BaseModelConfig):
    """Configuration for the ChannelMerger module.

    Parameters
    ----------
    embed_ref :
        Also embed the reference position, e.g. to enable handling bipolar channels. This requires
        passing both `positions` and `ref_positions` to `forward()`.
    unmerge :
        If True, unmerge (rather than merge) channels. This is useful to compute the inverse
        operation of a default `ChannelMerger`. In this case, the input to `forward()` should be of
        shape (B, n_virtual_channels, T).
    """

    name: tp.Literal["ChannelMerger"] = "ChannelMerger"

    n_virtual_channels: int = 270
    fourier_emb_config: FourierEmbConfig = FourierEmbConfig(
        n_freqs=None,
        total_dim=288,
        n_dims=2,
    )
    dropout: float = 0
    usage_penalty: float = 0.0
    n_subjects: int = 200
    per_subject: bool = False
    embed_ref: bool = False
    unmerge: bool = False

    def build(self) -> nn.Module:
        return ChannelMerger(self)


class ChannelMerger(nn.Module):
    """Module to merge (or unmerge) channels using channel attention based on channel coordinates."""

    def __init__(self, config: ChannelMergerConfig = ChannelMergerConfig()):
        super().__init__()
        self.embedding = config.fourier_emb_config.build()
        pos_dim = self.embedding.total_dim
        assert isinstance(pos_dim, int)  # for mypy

        self.per_subject = config.per_subject
        self.embed_ref = config.embed_ref
        n_params_pos_dim = pos_dim * 2 if self.embed_ref else pos_dim
        if self.per_subject:
            self.heads = nn.Parameter(
                torch.randn(
                    config.n_subjects, config.n_virtual_channels, n_params_pos_dim
                )
            )
        else:
            self.heads = nn.Parameter(
                torch.randn(config.n_virtual_channels, n_params_pos_dim)
            )
        self.invalid_value = ChannelPositions.INVALID_VALUE
        self.heads.data /= pos_dim**0.5  # XXX Double check
        self.dropout = config.dropout
        self.usage_penalty = config.usage_penalty
        self._penalty = torch.tensor(0.0)
        self.unmerge = config.unmerge

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def _get_weights(
        self,
        subject_ids: torch.Tensor,
        positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        B, C, _ = positions.shape
        if self.embed_ref:
            assert (
                positions.shape[2] == self.embedding.n_dims * 2  # type: ignore
            ), "Expecting 4 spatial dimensions when self.embed_ref is True"
            embedding = torch.cat(
                [
                    self.embedding(positions[..., : self.embedding.n_dims]),  # type: ignore
                    self.embedding(positions[..., self.embedding.n_dims :]),  # type: ignore
                ],
                dim=2,
            )
        else:
            assert positions.shape[2] == self.embedding.n_dims
            embedding = self.embedding(positions)

        score_offset = torch.zeros(B, C, device=device)
        invalid_mask = (positions == self.invalid_value).all(dim=-1)
        score_offset = score_offset.masked_fill(invalid_mask, float("-inf"))

        if self.training and self.dropout:
            if self.unmerge:
                raise NotImplementedError(
                    "Figure out how to apply dropout if unmerge=True"
                )
            center_to_ban = torch.rand(2, device=device)
            radius_to_ban = self.dropout
            banned = (positions[:, :, :2] - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset = score_offset.masked_fill(banned, float("-inf"))

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            heads = self.heads.gather(
                0, subject_ids.view(-1, 1, 1).expand(-1, cout, pos_dim)
            )
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        if self.unmerge:
            scores = scores.transpose(1, 2)
        return torch.softmax(scores, dim=2).nan_to_num()  # Replace nans by 0

    def forward(
        self,
        meg: torch.Tensor,
        subject_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spatial attention on input.

        Parameters
        ----------
        positions :
            Normalized (x, y) coordinates for each channel in `meg`, of shape (B, C, 2).
            If shape is (B, C, 4), the additional two coordinates per channel indicate the
            position of the reference electrode. See `ns.features.neuro.ChannelPositions`.
        """
        weights = self._get_weights(subject_ids, positions, meg.device)
        out = weights @ meg
        if self.training and self.usage_penalty > 0.0:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class UnitNorm(nn.Module):
    """Normalize last dimension of tensor to have unit Frobenius norm.

    Useful for parametrizing different normalization alternatives in `MlpConfig` below.

    NOTE: `hidden_dim` argument included for consistency with other normalization layers (e.g.
          BatchNorm).
    """

    def __init__(self, hidden_dim: int = 0) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p="fro", dim=-1, keepdim=True)


class MlpConfig(BaseModelConfig):
    """Multilayer perceptron, e.g. for use as projection head.

    Notes
    -----
    Input size can be specified in the config or at build time.
    Output size can either be specified in the config (as the last element of `hidden_sizes`) or at
    build time through the `output_size` parameter (like other models in neuraltrain), in which
    case this will overwrite the last value in `hidden_sizes`.
    For convenience, passing an empty list of hidden sizes yields `nn.Identity`.
    """

    name: tp.Literal["Mlp"] = "Mlp"

    input_size: int | None = None
    hidden_sizes: list[int]

    norm_layer: tp.Literal["layer", "batch", "instance", "unit", None] = None
    activation_layer: tp.Literal["relu", "gelu", "elu", "prelu", None] = "relu"

    bias: bool = True
    dropout: float = 0.0

    @staticmethod
    def _get_norm_layer(kind: str | None) -> tp.Type[nn.Module] | None:
        return {
            "batch": nn.BatchNorm1d,
            "layer": nn.LayerNorm,
            "instance": nn.InstanceNorm1d,
            "unit": UnitNorm,
            None: None,
        }[kind]

    @staticmethod
    def _get_activation_layer(kind: str | None) -> tp.Type[nn.Module]:
        return {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            None: nn.Identity,
        }[kind]

    def build(
        self, input_size: int | None = None, output_size: int | None = None
    ) -> nn.Sequential | nn.Identity:
        if not self.hidden_sizes:
            return nn.Identity()

        input_size = self.input_size if input_size is None else input_size
        assert input_size is not None, "input_size cannot be None."
        hidden_sizes = self.hidden_sizes
        if output_size is not None:
            hidden_sizes[-1] = output_size

        return MLP(
            in_channels=input_size,
            hidden_channels=hidden_sizes,
            norm_layer=self._get_norm_layer(self.norm_layer),
            activation_layer=self._get_activation_layer(self.activation_layer),
            bias=self.bias,
            dropout=self.dropout,
        )


