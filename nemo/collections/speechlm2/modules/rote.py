# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Time Embedding (ROTE).

ROTE encodes *absolute time* into audio feature embeddings. It is mechanically
identical to Rotary Position Embedding (RoPE), but the rotation angle is derived
from the **time index** of each frame (in seconds) instead of its token
position. Unlike RoPE in LLMs, ROTE is not applied to attention Q/K — it is
applied once to the (audio) feature embeddings, encoding time information into
the features directly instead of adding time as separate tokens.

This mirrors the single-time-axis formulation used by OmniVinci (``media_encoder.py``),
using the GPT-J / interleaved pairing convention (consecutive ``(2k, 2k+1)`` channels
form each rotation plane), with optional ``max_time`` normalization:

    inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))   # (dim/2,)
    times    = times / max_time * 2π                       # if max_time is set (OmniVinci)
    freqs    = times[..., None] * inv_freq                 # (..., T, dim/2)
    emb      = repeat_interleave(freqs, 2, dim=-1)         # (..., T, dim): [f0, f0, f1, f1, ...]
    out      = x * emb.cos() + rotate_half(x) * emb.sin()

Because the angle is a function of absolute time, a new audio segment simply
starts its time index from its own offset (0 for an independent audio), which
naturally "resets" the rotation for that segment.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent channel pairs (GPT-J / interleaved): ``[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]``.

    Matches the ``rotate_half`` used by OmniVinci, where each consecutive
    ``(2k, 2k+1)`` pair forms a rotation plane
    """
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class RotaryTimeEmbedding(nn.Module):
    """Apply rotary time embedding to feature embeddings.

    Args:
        dim: Feature dimension of the embeddings ROTE is applied to (the channel
            dimension ``C`` of the ``(B, T, C)`` input). Used to derive the
            rotary frequencies.
        theta: Base used for the geometric progression of inverse frequencies
            (a.k.a. ``rope_theta``). If ``None`` (default), it is derived from
            ``max_time`` as ``max_time / 2π`` (OmniVinci) when ``max_time`` is set,
            otherwise the standard RoPE default of ``10000.0``. Pass a value to
            override.
        rotary_fraction: Fraction of channels to rotate, in ``(0, 1]``. The first
            ``rot_dim`` channels (rounded down to an even number) are rotated and
            the remaining channels are passed through unchanged, matching the
            partial-rotary slicing used by the reference ``apply_rotary_emb``.
        max_time: Optional maximum expected time in seconds. When set, per-frame
            ``times`` are normalized as ``times / max_time * 2π`` before computing
            angles (the OmniVinci ``media_encoder`` scheme): the fastest channel
            then completes exactly one rotation over ``max_time`` and every slower
            channel less than one, giving an alias-free, monotonic phase across
            ``[0, max_time]``. When ``None`` (default), raw seconds are used as-is
            (standard RoPE: fast channels wrap, slow channels disambiguate).
    """

    def __init__(
        self, dim: int, theta: Optional[float] = None, rotary_fraction: float = 1.0, max_time: Optional[float] = None
    ):
        super().__init__()
        if not 0.0 < rotary_fraction <= 1.0:
            raise ValueError(f"rotary_fraction must be in (0, 1], got {rotary_fraction}.")
        if max_time is not None and max_time <= 0.0:
            raise ValueError(f"max_time must be positive, got {max_time}.")
        # Resolve theta. OmniVinci derives it from max_time (theta = max_time / 2π) so the slowest
        # channel spans ~max_time; combined with the forward-pass time normalization this spreads the
        # frequency bank across the clip. An explicit theta overrides; with no max_time we fall back
        # to the standard RoPE default of 10000.
        if theta is None:
            theta = (max_time / (2.0 * math.pi)) if max_time is not None else 10000.0
        self.dim = dim
        self.theta = theta
        self.rotary_fraction = rotary_fraction
        self.max_time = max_time
        # Number of channels actually rotated; must be even so it splits into pairs.
        rot_dim = int(dim * rotary_fraction)
        rot_dim -= rot_dim % 2
        if rot_dim < 2:
            raise ValueError(
                f"rotary_fraction={rotary_fraction} and dim={dim} yield rot_dim={rot_dim}; "
                "need at least 2 channels to rotate."
            )
        self.rot_dim = rot_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
        # Derived (not trained); recomputable from config, so keep out of the state dict.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        """Rotate ``x`` according to per-frame ``times``.

        Args:
            x: Feature embeddings of shape ``(B, T, C)`` (channel-last).
            times: Per-frame absolute time in seconds, shape ``(B, T)`` (or any
                shape broadcastable to ``x[..., 0]``).

        Returns:
            Tensor of the same shape and dtype as ``x``, with the first
            ``rot_dim`` channels rotated by the time-dependent angle and the rest
            passed through unchanged.
        """
        ori_dtype = x.dtype
        # Run the rotation in fp32 with autocast disabled, then cast back to the input dtype.
        # OmniVinci uses fp64 here, but fp32 (could use device dtype though) is ample for this angle math (cheaper on GPU): the
        # phase is bounded once max_time normalization maps times into [0, 2π].
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()           # maybe we can remove this
            x_rot, x_pass = x[..., : self.rot_dim], x[..., self.rot_dim :]

            times = times.float()   # and this?
            if self.max_time is not None:
                # OmniVinci normalization: map [0, max_time] -> [0, 2π] so the fastest channel
                # (inv_freq[0] == 1) completes exactly one rotation over max_time and slower
                # channels less — an alias-free, monotonic phase across the expected duration.
                times = times / self.max_time * (2.0 * math.pi)

            # angles: (..., T, rot_dim/2) -> (..., T, rot_dim)
            freqs = times.unsqueeze(-1) * self.inv_freq.to(device=x.device, dtype=torch.float32)
            # Interleave each frequency twice ([f0, f0, f1, f1, ...]) so it aligns with the
            # consecutive-pair rotation done by ``rotate_half`` (GPT-J / OmniVinci convention).
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            cos, sin = emb.cos(), emb.sin()

            out = torch.cat((x_rot * cos + rotate_half(x_rot) * sin, x_pass), dim=-1)
        return out.to(ori_dtype)
