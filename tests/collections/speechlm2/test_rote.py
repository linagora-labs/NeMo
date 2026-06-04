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
import pytest
import torch

from nemo.collections.speechlm2.modules.rote import RotaryTimeEmbedding, rotate_half
from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking


@pytest.mark.unit
def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Interleaved (GPT-J) convention: pairs (1, 2), (3, 4) -> (-2, 1), (-4, 3) = [-2, 1, -4, 3]
    assert torch.equal(rotate_half(x), torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


@pytest.mark.unit
def test_rote_zero_time_is_identity():
    rote = RotaryTimeEmbedding(dim=16)
    x = torch.randn(2, 5, 16)
    times = torch.zeros(2, 5)
    out = rote(x, times)
    torch.testing.assert_close(out, x)


@pytest.mark.unit
def test_rote_is_norm_preserving():
    rote = RotaryTimeEmbedding(dim=32)
    x = torch.randn(3, 7, 32)
    times = torch.rand(3, 7) * 12.3
    out = rote(x, times)
    # Rotation preserves the L2 norm of each frame vector.
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


@pytest.mark.unit
def test_rote_relative_time_property():
    """The inner product between two frames depends only on their time *difference*."""
    rote = RotaryTimeEmbedding(dim=64)
    a = torch.randn(1, 1, 64)
    b = torch.randn(1, 1, 64)
    delta = 0.37

    def dot_at(t0):
        times = torch.tensor([[t0, t0 + delta]])
        rotated = rote(torch.cat([a, b], dim=1), times)
        return (rotated[0, 0] * rotated[0, 1]).sum()

    # Same delta at different absolute offsets must give the same inner product.
    torch.testing.assert_close(dot_at(0.0), dot_at(5.0))
    torch.testing.assert_close(dot_at(0.0), dot_at(100.0))


@pytest.mark.unit
def test_rote_partial_rotary_leaves_tail_untouched():
    rote = RotaryTimeEmbedding(dim=16, rotary_fraction=0.5)
    assert rote.rot_dim == 8
    x = torch.randn(2, 4, 16)
    times = torch.rand(2, 4) * 3.0
    out = rote(x, times)
    # Tail channels (beyond rot_dim) pass through unchanged.
    torch.testing.assert_close(out[..., rote.rot_dim :], x[..., rote.rot_dim :])
    # Rotated channels do change for non-zero time.
    assert not torch.allclose(out[..., : rote.rot_dim], x[..., : rote.rot_dim])


@pytest.mark.unit
def test_rote_matches_reference_formula():
    """Numerical parity with the MusicFlamingo / OmniVinci single-time-axis reference snippet."""
    dim, theta = 16, 10000.0
    rote = RotaryTimeEmbedding(dim=dim, theta=theta)
    x = torch.randn(2, 6, dim)
    times = torch.rand(2, 6) * 9.0

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = times.unsqueeze(-1) * inv_freq  # (B, T, dim/2)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)  # (B, T, dim): interleaved [f0, f0, f1, f1, ...]
    ref = x * emb.cos() + rotate_half(x) * emb.sin()

    torch.testing.assert_close(rote(x, times), ref)


@pytest.mark.unit
def test_rote_max_time_normalization():
    """With ``max_time`` set, times are normalized to [0, 2π] (OmniVinci scheme).

    Equivalent to running the un-normalized module on ``times / max_time * 2π``.
    """
    import math

    dim, max_time = 16, 30.0
    # Pin theta so both modules share the same frequency bank (max_time would otherwise derive it).
    rote_norm = RotaryTimeEmbedding(dim=dim, theta=10000.0, max_time=max_time)
    rote_raw = RotaryTimeEmbedding(dim=dim, theta=10000.0)
    x = torch.randn(2, 5, dim)
    times = torch.rand(2, 5) * max_time

    out_norm = rote_norm(x, times)
    out_raw = rote_raw(x, times / max_time * (2.0 * math.pi))
    torch.testing.assert_close(out_norm, out_raw)


@pytest.mark.unit
def test_rote_max_time_derives_theta():
    """When ``max_time`` is set and ``theta`` is omitted, theta is derived as max_time / 2π."""
    import math

    max_time = 30.0
    rote = RotaryTimeEmbedding(dim=16, max_time=max_time)
    assert rote.theta == pytest.approx(max_time / (2.0 * math.pi))
    # Explicit theta overrides the derivation.
    assert RotaryTimeEmbedding(dim=16, theta=10000.0, max_time=max_time).theta == 10000.0


@pytest.mark.unit
def test_rote_invalid_rotary_fraction():
    with pytest.raises(ValueError):
        RotaryTimeEmbedding(dim=16, rotary_fraction=0.0)
    with pytest.raises(ValueError):
        RotaryTimeEmbedding(dim=16, rotary_fraction=1.5)
    # dim=2 with fraction too small -> rot_dim < 2.
    with pytest.raises(ValueError):
        RotaryTimeEmbedding(dim=2, rotary_fraction=0.4)


class _RotePerception(torch.nn.Module):
    """Minimal perception stub that applies ROTE at the encoder output.

    Mirrors ``AudioPerceptionModule``: identity "encoder" producing one feature frame per
    input sample, ROTE applied at the modality-adapter entrance using ``time_offset``.
    """

    def __init__(self, dim, frame_duration):
        super().__init__()
        self.rote = RotaryTimeEmbedding(dim=dim)
        self._frame_duration = frame_duration
        self._dim = dim
        # Fixed projection so each sample maps to a deterministic feature vector.
        gen = torch.Generator().manual_seed(0)
        self.codebook = torch.randn(256, dim, generator=gen)

    def forward(self, input_signal=None, input_signal_length=None, time_offset=None):
        # input_signal: (B, T_samples); treat each sample as one encoder frame.
        b, t = input_signal.shape
        feats = self.codebook[input_signal.long().clamp(0, 255)]  # (B, T, dim)
        frame_times = (torch.arange(t, dtype=torch.float32) + 0.5) * self._frame_duration
        times = frame_times.unsqueeze(0).expand(b, -1)
        if time_offset is not None:
            times = times + time_offset.float().unsqueeze(1)
        feats = self.rote(feats, times)
        return feats, input_signal_length.clone()


@pytest.mark.unit
def test_rote_chunking_time_continuity():
    """Encoding one audio whole vs. chunked must yield identical ROTE'd embeddings.

    This is the key correctness check for the chunking-aware ``time_offset``: without it,
    each chunk's ROTE time would restart at 0 and the recombined embeddings would differ.
    """
    sampling_rate = 1  # 1 sample == 1 second, so frame index == absolute time index
    perception = _RotePerception(dim=16, frame_duration=1.0)
    audio = torch.arange(1, 13, dtype=torch.float32).unsqueeze(0)  # (1, 12)
    audio_lens = torch.tensor([12], dtype=torch.long)

    whole = encode_audio_with_optional_chunking(
        perception, audio, audio_lens, chunk_size_seconds=None, sampling_rate=sampling_rate
    )
    chunked = encode_audio_with_optional_chunking(
        perception, audio, audio_lens, chunk_size_seconds=4.0, sampling_rate=sampling_rate
    )

    assert len(whole) == len(chunked) == 1
    assert whole[0].shape == chunked[0].shape == (12, 16)
    torch.testing.assert_close(whole[0], chunked[0])


@pytest.mark.unit
def test_rote_second_audio_resets_time():
    """Two independent audios (separate rows) each start ROTE time at 0."""
    perception = _RotePerception(dim=16, frame_duration=1.0)
    # Identical content in two rows -> identical ROTE output (both start at t=0).
    audio = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    audio_lens = torch.tensor([3, 3], dtype=torch.long)
    embs = encode_audio_with_optional_chunking(perception, audio, audio_lens, chunk_size_seconds=None, sampling_rate=1)
    assert len(embs) == 2
    torch.testing.assert_close(embs[0], embs[1])
