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
import torch


class SpecialRowsHead(torch.nn.Module):
    """Frozen LM head plus trainable additive deltas on a few vocabulary rows.

    Duplex models spend most frames emitting ``<pad>`` (silence) and must learn when to emit
    BOS/EOS, tokens a text LLM rarely or never predicted. With tied embeddings whose ``<pad>``
    row is ~0 (e.g. Luciole), training the whole head with Adam lowers every non-target row
    and the ~0 ``<pad>`` logit wins on every frame, while a frozen head never learns to emit
    ``<pad>``. Training only these rows avoids both failure modes.

    Args:
        base: The pretrained LM head (possibly tied to the input embeddings). It is frozen.
        ids: Token IDs whose output rows get a trainable delta.
    """

    def __init__(self, base: torch.nn.Linear, ids: list[int]):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.delta = torch.nn.Parameter(torch.zeros(len(ids), base.in_features, dtype=base.weight.dtype))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden)
        return logits.index_add(-1, self.ids, hidden @ self.delta.T.to(hidden.dtype))
