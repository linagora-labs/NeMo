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
import re
import warnings
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import _resolve_audios_in_prompt, replace_placeholders_and_build_targets
from nemo.collections.speechlm2.parts.automodel_lora import ensure_lora_trainable, make_peft_config, maybe_install_lora
from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.mtp import (
    build_mtp_loss_fn,
    calculate_mtp_loss_with_per_depth,
    calculate_mtp_teacher_forced_agreement,
    compute_mtp_agreement_lengths,
    mtp_validation_forward,
    vocab_parallel_argmax,
)
from nemo.collections.speechlm2.parts.multispeaker import build_speaker_tokens, maybe_init_lss_loss
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    _resolve_llm_config_overrides,
    load_pretrained_automodel_llm,
    maybe_load_pretrained_models,
    setup_speech_encoder,
    update_perception_output_dim,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.core.utils.lightning_utils import read_batch
from nemo.utils import logging, logging_mode


class SALMAutomodel(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALMAutomodel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        tokenizer_src = self.cfg.get("tokenizer_path", None) or self.cfg.pretrained_llm
        self.tokenizer = AutoTokenizer(
            tokenizer_src, use_fast=True, trust_remote_code=self.cfg.get("trust_remote_code", False)
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.speaker_token_ids = build_speaker_tokens(self.cfg.get("speaker_tokens", None), self.tokenizer)
        self.lss_loss = maybe_init_lss_loss(self.cfg.get("lss_loss", None), self.speaker_token_ids)
        self.llm = None  # populated by configure_model
        self.perception = None  # populated by configure_model

        self._use_fsdp = False
        self._use_tp = False

        # Fused CE + per-bucket memory profiling state (see _setup_fused_cross_entropy
        # and _profile_step_memory). Set here so inference paths that never call
        # on_fit_start still find them defined.
        self._fused_ce = None
        self._bucket_memory: dict[int, tuple[float, float, int, int, int]] = {}
        self._prev_step_shape: tuple[int, int] | None = None
        self._peak_alloc_gib = 0.0
        self._memory_snapshot_active = False

        if self.cfg.get("init_configure_model", False):
            self.configure_model()

    @property
    def device(self) -> torch.device:
        """Infer device from the LLM's parameters.

        ``LightningModule.device`` is set by the Trainer and defaults to CPU
        during standalone inference (no Trainer).  Override to query the actual
        parameter storage so that ``.to(self.device)`` works correctly for
        both regular and DTensor (FSDP2/distributed) parameters.
        """
        if self.llm is not None:
            p = next(self.llm.parameters(), None)
            if p is not None:
                return p._local_tensor.device if isinstance(p, DTensor) else p.device
        return super().device

    @property
    def _mtp_enabled(self) -> bool:
        """True when the MTP head is attached, regardless of how the model was loaded."""
        return getattr(getattr(self, "llm", None), "mtp", None) is not None

    @property
    def _mtp_num_depths(self) -> int:
        """Return the logical number of MTP prediction depths."""
        if not self._mtp_enabled:
            return 0
        mtp_config = getattr(self.llm, "mtp_config", None)
        if mtp_config is None:
            raise RuntimeError("The attached MTP head does not expose its logical depth through llm.mtp_config")
        return int(mtp_config.num_layers)

    @property
    def _context_parallel_size(self) -> int:
        """Return the configured context-parallel world size."""
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None or "cp" not in (device_mesh.mesh_dim_names or ()):
            return 1
        return int(device_mesh["cp"].size())

    @property
    def embed_tokens(self):
        """Navigate to the LLM's embedding layer (kept inside the LLM).

        Use the HF `get_input_embeddings()` API rather than a hardcoded
        `self.llm.model.embed_tokens` path: the latter is Llama/Qwen-specific
        and breaks on architectures that name their trunk/embedding differently
        (e.g. dense Nemotron-H, whose embedding lives at `backbone.embeddings`).
        This matches the non-automodel SALM (salm.py). Returns the nn.Embedding
        module, so `.weight` / `.num_embeddings` keep working.
        """
        if self.llm is None:
            return None
        return self.llm.get_input_embeddings()

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the LLM's embedding table.

        Uses ``F.embedding`` instead of calling the ``nn.Embedding`` module to
        avoid triggering FSDP2 pre-forward hooks (which lazily initialize the
        child before the root LLM module, causing a ``RuntimeError``).

        When the weight is a sharded ``DTensor`` (FSDP2), we ``full_tensor()``
        it first to all-gather the complete embedding table — the same operation
        FSDP2 performs inside the LLM's forward pass.
        """
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        return torch.nn.functional.embedding(input_ids, weight)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
        **llm_kwargs,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        ``llm_kwargs`` carries optional THD/packed-sequence metadata
        (``qkv_format``, ``cu_seqlens``, ``position_ids``, ``max_seqlen``)
        and CP-prepared MTP position IDs. ``mtp_embed_inputs`` is removed and
        passed through Automodel's positional per-depth embedding contract.
        These values are absent for the BSHD path.
        """
        # input_embeds: (B, T, H) for BSHD or (T_total, H) for THD packed
        # (the THD shape mirrors Automodel's _shard_thd_chunk_for_te output —
        # the model squeezes 3D inputs internally when qkv_format=="thd", so
        # passing 2D directly skips that hop)
        llm_input_ids = None
        if llm_kwargs.get("qkv_format") == "thd":
            # Automodel's THD preprocessor still squeezes input_ids even when
            # inputs_embeds carries the real token/audio embeddings.
            seq_len = input_embeds.shape[0] if input_embeds.ndim == 2 else input_embeds.shape[1]
            llm_input_ids = torch.zeros((1, seq_len), device=input_embeds.device, dtype=torch.long)

        mtp_embed_inputs = tuple(llm_kwargs.pop("mtp_embed_inputs", ()))
        out = self.llm(
            llm_input_ids,
            *mtp_embed_inputs,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
            **llm_kwargs,
        )
        if not isinstance(out, dict):
            # NeMo Automodel doesn't respect return_dict=True yet
            ans = {"logits": out}
        else:
            ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
            if cache is not None:
                ans["cache"] = out["past_key_values"]
            # MTP per-depth hidden states are returned when an MTP head is attached and
            # MTP computation is enabled for this forward (during training or explicitly
            # during validation).
            mtp_h = getattr(out, "mtp_per_depth_h", None)
            if mtp_h is not None:
                ans["mtp_per_depth_h"] = mtp_h
        return ans

    def _uses_parallel_expert_encoder(self) -> bool:
        """Whether the mounted perception encoder is a ``ParallelExpertEncoder``.

        The PE encoder performs its own context-preserving long-form online inference
        (``forward`` -> ``_forward_online``), so audio must be fed to it as a
        single long sequence when oracle speaker targets are absent.
        """
        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

        return isinstance(getattr(self.perception, "encoder", None), ParallelExpertEncoder)

    def _warn_parallel_expert_encoder_inference_compatibility(self, cp_size: int) -> None:
        if not self.cfg.get("pe_encoder_path", None):
            return

        unsupported = []
        if self.cfg.get("encoder_chunk_size_seconds", None) is not None:
            unsupported.append("encoder_chunk_size_seconds")
        if self.cfg.get("packed_sequences", False):
            unsupported.append("packed_sequences")
        if cp_size > 1:
            unsupported.append(f"cp_size={cp_size}")

        if unsupported:
            warnings.warn(
                "The ParallelExpertEncoder inference path is currently experimental and does not support "
                f"{', '.join(unsupported)}. It will be made to work with these options later."
                " This warning only applies when `spk_targets` are absent; training with ground-truth "
                "`spk_targets` still uses the regular encoder chunking path.",
                stacklevel=2,
            )

    def prepare_inputs(self, batch: dict, *, include_mtp_inputs: bool = True):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations. With a
          ``ParallelExpertEncoder`` the full audio is encoded in a single
          perception forward (long-form streaming is handled inside the encoder).
          With an ordinary encoder, long source audio is optionally time-chunked
          and recombined via ``parts.encoder_chunking``.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.

        When ``batch["spk_targets"]`` is present, those RTTM-derived speaker
        targets are injected into a ``ParallelExpertEncoder``. Otherwise, the
        encoder runs its embedded Sortformer to predict diarization.

        ``include_mtp_inputs=False`` avoids constructing future-token tensors
        when validation cannot consume MTP outputs, such as under CP.
        """
        from nemo.collections.speechlm2.parts.cp_helpers import (
            encode_audio_with_cp_distribution,
            get_cp_mesh,
            get_perception_fsdp_group,
        )

        device_mesh = getattr(self, "_device_mesh", None)
        spk_targets = batch.get("spk_targets", None)
        cp_mesh, cp_size, _ = get_cp_mesh(device_mesh)
        fsdp_sync_group = get_perception_fsdp_group(device_mesh)

        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        # Encoder path by (PEE, spk_targets):
        # PEE=true  & spk_targets=None  : Inference mode, uses recursive encoding in PEE, NO chunking/CP.
        # PEE=true  & spk_targets!=None : Training mode, ``spk_targets`` injected into PEE with chunking/CP.
        # PEE=false & spk_targets=None  : Training/Inference mode, plain encoder with chunking/CP.
        # PEE=false & spk_targets!=None : Training/Inference mode, plain encoder with chunking/CP and
        #                                 the provided ``spk_targets`` is ignored (no-op).
        dummy_audio_loss = None
        if self._uses_parallel_expert_encoder() and spk_targets is None:
            self._warn_parallel_expert_encoder_inference_compatibility(cp_size)
            audio_embs, audio_emb_lens = self.perception(
                input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
            )
            audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
        else:
            audio_embs, dummy_audio_loss = encode_audio_with_cp_distribution(
                self.perception,
                batch["audios"],
                batch["audio_lens"],
                chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                sampling_rate=self.sampling_rate,
                cp_mesh=cp_mesh,
                spk_targets=spk_targets,
                fsdp_sync_group=fsdp_sync_group,
                return_dummy_loss=True,
            )
        input_ids_to_embed = torch.where(batch["input_ids"] == self.audio_locator_tag_id, 0, batch["input_ids"])
        text_embs = self._embed_tokens(input_ids_to_embed)
        target_ids_full = batch["input_ids"].where(batch["loss_mask"], -100)  # CrossEntropyLoss().ignore_index

        # Packed-sequence (THD) path — used for both training and validation when enabled.
        # Generate stays on the BSHD path (it doesn't go through prepare_inputs).
        if self.cfg.get("packed_sequences", False):
            from nemo.collections.speechlm2.parts.packed_sequences import prepare_packed_llm_inputs

            ans = prepare_packed_llm_inputs(
                input_ids=batch["input_ids"],
                text_embs=text_embs,
                audio_embs=audio_embs,
                target_ids=target_ids_full,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                device_mesh=device_mesh,
                mtp_num_depths=self._mtp_num_depths if include_mtp_inputs else 0,
            )
            if dummy_audio_loss is not None:
                ans["dummy_audio_loss"] = dummy_audio_loss
            return ans

        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=target_ids_full,
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # BSHD path runs only when CP is inactive (the fit-start validator
        # rejects BSHD + CP > 1, see _validate_parallelism_compatibility).
        # Truncate the seq dim to be divisible by tp_size so sequence
        # parallelism doesn't reshape the input under us.
        if self._use_tp:
            tp_size = self.device_mesh["tp"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_size) != 0:
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        ans = {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "llm_kwargs": {},
        }
        if dummy_audio_loss is not None:
            ans["dummy_audio_loss"] = dummy_audio_loss
        return ans

    def on_fit_start(self) -> None:
        """Configure the MoE aux-loss backward scaler to cancel FSDP's gradient
        averaging (see ``_configure_moe_aux_loss_scaler``)."""
        self._validate_parallelism_compatibility()
        self._configure_moe_aux_loss_scaler()
        self._apply_train_eval_modes()
        self._report_activation_checkpointing()
        self._setup_fused_cross_entropy()

    def _llm_trunk(self):
        """The LLM's transformer trunk -- the module holding ``.layers``.

        Its name differs between implementations (transformers' built-in nemotron_h calls
        it ``model``, the bundled remote code calls it ``backbone``), and a PEFT wrapper
        may add a level of attribute forwarding on top. Probing for ``.layers`` identifies
        the trunk regardless of which of those is in play. Returns ``None`` if no candidate
        matches, so callers can fall back instead of crashing.
        """
        for attr in ("model", "backbone"):
            trunk = getattr(self.llm, attr, None)
            if trunk is not None and getattr(trunk, "layers", None) is not None:
                return trunk
        return None

    def _llm_decoder_layers(self) -> list:
        """The LLM's decoder layer list, under either trunk name (`model` or `backbone`)."""
        trunk = self._llm_trunk()
        return list(trunk.layers) if trunk is not None else []

    def _report_activation_checkpointing(self) -> None:
        """Log whether the requested LLM activation checkpointing is actually in effect.

        ``activation_checkpointing_llm: true`` can be honored all the way down to
        ``gradient_checkpointing_enable()`` and still be a no-op at forward time (see
        :meth:`_apply_train_eval_modes`). The failure costs tens of GB and reports nothing:
        no warning, no missing key, just an OOM some buckets later. So state the outcome
        explicitly at fit start rather than leaving it to be inferred from a crash.
        """
        strategy = getattr(self, "_trainer", None) and self._trainer.strategy
        if not getattr(strategy, "activation_checkpointing_llm", False):
            return
        layers = self._llm_decoder_layers()
        if not layers:
            warnings.warn(
                "activation_checkpointing_llm=true but no LLM decoder layers were found "
                f"under {type(self.llm).__name__}.model/.backbone -- cannot verify it is active."
            )
            return
        layer = layers[0]
        wrapped = any("checkpoint" in type(m).__name__.lower() for m in layer.modules())
        hf_native = bool(getattr(layer, "gradient_checkpointing", False))
        if wrapped or (hf_native and layer.training):
            logging.info(
                "LLM activation checkpointing ACTIVE (torch wrapper=%s, hf_native=%s, training=%s)",
                wrapped,
                hf_native,
                layer.training,
            )
        else:
            warnings.warn(
                "activation_checkpointing_llm=true but activation checkpointing is INERT on "
                f"{type(layer).__name__}: no checkpoint wrapper, hf_native={hf_native}, "
                f"training={layer.training}. Activations will be retained for every layer and "
                "long sequences will OOM. See SALMAutomodel._apply_train_eval_modes."
            )

    def on_validation_start(self) -> None:
        """Reject unsupported parallel layouts for fit and standalone validation."""
        self._validate_parallelism_compatibility(check_backward=False)

    def on_test_start(self) -> None:
        """Reject unsupported parallel layouts for standalone testing."""
        self._validate_parallelism_compatibility(check_backward=False)

    def _setup_fused_cross_entropy(self) -> None:
        """Wire up the fused linear cross-entropy, or explain why it stayed off.

        The plain path materializes three tensors of shape ``(B*T, vocab)``: the logits,
        the ``log_softmax`` saved for backward, and the incoming gradient. On Luciole-23B
        (vocab 128k) at ~9k tokens that is 3 x 2.3 GiB in bf16 -- the single largest
        identified block of the step. ``linear_cross_entropy`` fuses the ``lm_head`` matmul
        into the loss and streams it in tiles, so none of the three is ever allocated.

        Opt-in via ``model.fused_linear_cross_entropy: true``. Every prerequisite is checked
        here rather than at the first training step, so a misconfiguration surfaces at fit
        start with a reason instead of a traceback 20 minutes in -- and always degrades to
        the plain path rather than failing the run.
        """
        self._fused_ce = None
        if not self.cfg.get("fused_linear_cross_entropy", False):
            return

        def _off(reason: str) -> None:
            warnings.warn(f"fused_linear_cross_entropy=true but staying on the plain logits path: {reason}")

        from nemo_automodel.components.loss.linear_ce import HAVE_CUT_CROSS_ENTROPY, FusedLinearCrossEntropy

        if self.lss_loss is not None:
            _off("latent speaker supervision (lss_loss) needs the full logits, which the fused path never materializes.")
            return
        if not HAVE_CUT_CROSS_ENTROPY:
            _off("the `cut_cross_entropy` package is not installed in this environment.")
            return
        # loss_parallel() shards the vocab dim across TP ranks; the fused kernel takes an
        # unsharded lm_head and would silently compute a different loss.
        if self._use_tp:
            _off("tensor parallelism is enabled, which needs the loss_parallel() vocab sharding.")
            return
        if self._llm_trunk() is None:
            _off(f"no transformer trunk with `.layers` found under {type(self.llm).__name__}.model/.backbone.")
            return
        if self._lm_head_weight() is None:
            _off("lm_head.weight could not be located on the LLM.")
            return
        # Without logits_to_keep the wrapper projects every position through lm_head, which
        # is the (B*T, vocab) allocation this whole path exists to avoid -- so the fused
        # loss would cost memory instead of saving it.
        import inspect

        try:
            params = inspect.signature(type(self.llm).forward).parameters
        except (TypeError, ValueError):
            params = {}
        if "logits_to_keep" not in params and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            _off(f"{type(self.llm).__name__}.forward does not accept `logits_to_keep`.")
            return

        self._fused_ce = FusedLinearCrossEntropy(ignore_index=-100, reduction="sum")
        logging.info("Fused linear cross-entropy ACTIVE (logits are never materialized).")

    def _lm_head_weight(self):
        """The output-projection weight, or ``None`` if it cannot be located."""
        get_out = getattr(self.llm, "get_output_embeddings", None)
        if callable(get_out):
            out = get_out()
            weight = getattr(out, "weight", None)
            if weight is not None:
                return weight
        for name, param in self.llm.named_parameters(remove_duplicate=False):
            if "lm_head" in name and name.endswith(".weight"):
                return param
        return None

    def _forward_hidden_states(self, input_embeds: Tensor, attention_mask: Tensor = None, **llm_kwargs) -> Tensor:
        """Final hidden states, without ever materializing full-vocabulary logits.

        The causal-LM wrapper has to stay the entry point. ``fully_shard`` registered the
        root FSDP unit on ``self.llm``, and only that unit's pre-forward hook converts the
        trunk's final ``norm`` weight (and ``lm_head``) from DTensor shards into plain
        tensors. Calling the trunk directly skips that hook and dies inside ``norm`` with
        "got mixed torch.Tensor and DTensor" -- the decoder layers survive it because each
        one is its own FSDP unit, but the root-level parameters are not.

        So call the wrapper and neutralize the expensive half instead: ``logits_to_keep=1``
        narrows ``lm_head`` to a single position (``slice(-1, None)``), and a forward hook
        on the trunk captures the hidden states the fused loss actually consumes.
        """
        captured = {}

        def _capture(_module, _args, output):
            captured["hidden"] = output["last_hidden_state"] if isinstance(output, dict) else output.last_hidden_state

        handle = self._llm_trunk().register_forward_hook(_capture)
        try:
            self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
                **llm_kwargs,
            )
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError("fused cross-entropy: the LLM trunk forward hook did not fire.")
        return captured.pop("hidden")

    def _fused_loss_sum(self, hidden: Tensor, target_ids: Tensor) -> Tensor:
        """Summed CE over the batch, computed without materializing logits."""
        lm_weight = self._lm_head_weight()
        # FSDP2 leaves lm_head as a DTensor shard; the kernel needs the full (V, H) matrix.
        # The plain path pays this same all-gather inside lm_head's forward, so it is not
        # extra memory -- it just becomes explicit here.
        if hasattr(lm_weight, "full_tensor"):
            lm_weight = lm_weight.full_tensor()
        return self._fused_ce(
            hidden.reshape(-1, hidden.size(-1)),
            target_ids.reshape(-1),
            lm_weight,
        )

    def _profile_step_memory(self, batch_size: int, seq_len: int) -> None:
        """Attribute the previous step's peak CUDA memory to the bucket shape that caused it.

        All 27 buckets carry a comparable token budget but not a comparable memory cost:
        ``(B=103, T=87)`` and ``(B=1, T=8416)`` hold ~9k tokens each yet load the audio
        encoder and the attention path completely differently. The run's ceiling is set by
        whichever single bucket peaks highest, so knowing *which* one lets us trim just that
        entry of ``bucket_batch_size`` instead of lowering the global token plateau.

        Peak stats are read at the start of a step because they then cover the whole of the
        previous step (forward, backward and optimizer), and are attributed to the shape
        recorded for that step rather than the current one.
        """
        if not torch.cuda.is_available():
            return
        prev = self._prev_step_shape
        if prev is not None:
            batch, seq = prev
            alloc = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            # Keyed by batch size, NOT by (batch, seqlen): ``bucket_batch_size`` assigns a
            # distinct batch size to each bucket, while the padded seqlen wanders inside a
            # bucket from one step to the next. Keying on the pair would grow one row per
            # step and make the table useless; keying on the batch size keeps it at one row
            # per bucket, and the seqlen of the worst step is carried along so the offending
            # batch stays identifiable.
            best = self._bucket_memory.get(batch)
            steps = best[4] + 1 if best is not None else 1
            if best is None or alloc > best[0]:
                self._bucket_memory[batch] = (alloc, reserved, seq, batch * seq, steps)
            else:
                self._bucket_memory[batch] = (best[0], best[1], best[2], best[3], steps)
            if alloc > self._peak_alloc_gib:
                self._peak_alloc_gib = alloc
                logging.info(
                    "[mem] new peak allocated: B=%d T=%d tokens=%d alloc=%.2f GiB reserved=%.2f GiB",
                    batch,
                    seq,
                    batch * seq,
                    alloc,
                    reserved,
                )
        torch.cuda.reset_peak_memory_stats()
        self._prev_step_shape = (int(batch_size), int(seq_len))

    def _dump_bucket_memory(self) -> None:
        """Log the per-bucket peak table, most expensive first."""
        if not self._bucket_memory:
            return
        rows = sorted(self._bucket_memory.items(), key=lambda kv: -kv[1][0])
        lines = ["[mem] peak allocated per bucket (worst step of each), most expensive first:"]
        lines.append("[mem]   batch    seqlen    tokens   alloc_GiB   reserved_GiB   steps")
        for batch, (alloc, reserved, seq, tokens, steps) in rows:
            lines.append(
                f"[mem] {batch:>7} {seq:>9} {tokens:>9} {alloc:>11.2f} {reserved:>14.2f} {steps:>7}"
            )
        logging.info("\n".join(lines))

    def _maybe_snapshot_memory(self, batch_idx: int) -> None:
        """Record a CUDA allocator trace over a few steps and dump it to a pickle.

        The per-bucket table says *how much* a step costs but not *what* holds it. This
        turns on ``torch.cuda.memory._record_memory_history``, which stamps every alloc
        and free with its Python stack, so the dump can be opened at
        https://docs.pytorch.org/memory_viz and read as "which tensor, allocated from
        which line, was alive at the peak".

        Opt-in and deliberately short-lived: the tracer adds per-allocation overhead and
        the pickle grows with the number of events, so it covers
        ``memory_snapshot_steps`` steps starting at ``memory_snapshot_at`` and then turns
        itself off. Rank 0 only -- every rank holds the same shapes here, and one file is
        enough.
        """
        at = int(self.cfg.get("memory_snapshot_at", 0))
        if at <= 0 or not torch.cuda.is_available():
            return
        span = int(self.cfg.get("memory_snapshot_steps", 3))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return

        if batch_idx == at:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            self._memory_snapshot_active = True
            logging.info("[mem] memory history recording ON at step %d (%d steps)", at, span)
        elif self._memory_snapshot_active and batch_idx >= at + span:
            import os

            out_dir = self.cfg.get("memory_snapshot_dir", None) or os.getcwd()
            path = os.path.join(out_dir, f"memory_snapshot_step{batch_idx}.pickle")
            try:
                torch.cuda.memory._dump_snapshot(path)
                logging.info("[mem] memory snapshot written to %s", path)
            except Exception as e:  # noqa: BLE001 -- profiling must never kill a run
                logging.warning("[mem] could not write the memory snapshot: %s", e)
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
                self._memory_snapshot_active = False

    def _validate_parallelism_compatibility(self, *, check_backward: bool = True) -> None:
        """Raise on known-incompatible THD/CP/backend configurations.

        Delegates to :func:`nemo.collections.speechlm2.parts.parallel.validate_parallelism_compatibility`
        with the runtime-derived values from this model's config and device mesh.
        """
        import os

        from nemo.collections.speechlm2.parts.parallel import validate_parallelism_compatibility

        cp_size = 1
        tp_size = 1
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is not None:
            names = device_mesh.mesh_dim_names or ()
            if "cp" in names:
                cp_size = device_mesh["cp"].size()
            if "tp" in names:
                tp_size = device_mesh["tp"].size()

        attn_backend = self.cfg.get("automodel_backend", {}).get("attn", "te")
        nvte_fused_attn = os.environ.get("NVTE_FUSED_ATTN")
        device_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

        validate_parallelism_compatibility(
            packed_sequences=bool(self.cfg.get("packed_sequences", False)),
            cp_size=cp_size,
            attn_backend=attn_backend,
            nvte_fused_attn=nvte_fused_attn,
            device_capability=device_capability,
            check_backward=check_backward,
        )
        mtp_cfg = self.cfg.get("mtp", None)
        mtp_enabled = mtp_cfg is not None and bool(mtp_cfg.get("enabled", False))
        if tp_size > 1 and mtp_enabled:
            raise ValueError(
                "SALMAutomodel MTP currently requires tp_size=1. The MTP head has no tensor-parallel plan, "
                "and fused MTP loss cannot materialize a TP-sharded LM-head weight with a DP-only "
                "gradient-reduction group."
            )

    def training_step(self, dataloader_iter):
        # ``dataloader_iter`` signature → Lightning selects
        # ``_DataLoaderIterDataFetcher`` (no prefetch) which is required for
        # bit-identical checkpoint resumption. See ``read_batch`` docstring.
        batch, batch_idx = read_batch(dataloader_iter, self)
        return self._training_step_batch(batch, batch_idx)

    def _apply_train_eval_modes(self) -> None:
        """Put frozen submodules in eval and every trainable one back in train mode.

        The train/eval flag is NOT cosmetic on the LLM. transformers' HF-native activation
        checkpointing is guarded by ``if self.gradient_checkpointing and self.training``
        (``modeling_layers.py::GradientCheckpointingLayer.__call__``), so a LLM left in eval
        SILENTLY skips checkpointing even with ``activation_checkpointing_llm: true`` -- the
        flag is set, ``gradient_checkpointing_enable()`` is called, and every layer still
        falls through to the plain forward. ``from_pretrained`` returns the model in eval and
        nothing restored it, so the LLM stayed there for the whole run.

        That is what OOM'd Luciole-23B: ~2 GB of retained activations per layer instead of
        the ~100 MB a checkpointed layer stores, i.e. ~60 GB of activations on a forward that
        should have used 4. The 8B never hit it -- ``NemotronHForCausalLM`` has a registered
        parallelization strategy that wraps layers with torch's ``checkpoint_wrapper``, which
        ignores train/eval; only models falling back to the default strategy (dense
        ``nemotron``, i.e. the 23B) get the mode-dependent HF-native path.

        Setting train mode here is also simply correct: a LoRA'd LLM has trainable params, so
        it belongs in train mode. Dropout is 0.0 in these recipes and Nemotron uses LayerNorm
        (mode-independent), so nothing else changes.
        """
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            m.eval() if is_frozen(m) else m.train()

    def _training_step_batch(self, batch: dict, batch_idx: int):
        self._current_batch_idx = batch_idx
        self._apply_train_eval_modes()
        # Before the forward, so the trace covers whole steps end to end.
        self._maybe_snapshot_memory(batch_idx)

        inputs = self.prepare_inputs(batch)
        self._record_training_stats(batch, inputs)
        if self._fused_ce is not None:
            forward_outputs = {
                "hidden_states": self._forward_hidden_states(
                    inputs["input_embeds"],
                    attention_mask=inputs["attention_mask"],
                    **inputs.get("llm_kwargs", {}),
                )
            }
        else:
            forward_outputs = self(
                inputs["input_embeds"],
                attention_mask=inputs["attention_mask"],
                **inputs.get("llm_kwargs", {}),
            )
        num_frames = (inputs["target_ids"] != -100).long().sum()

        # Match Automodel's training recipe: normalize CE by the *global* token count across
        # the DP group rather than each rank's local count. With variable-length speech batches
        # a local normalizer makes every rank contribute a differently-scaled gradient, and
        # FSDP's gradient averaging doesn't recover the true global mean. All-reduce the
        # labeled-token count and scale the per-rank loss by ``dp_size`` so that FSDP's
        # gradient averaging yields ``sum(rank_CE_sum) / num_frames_global``.
        dp_group = self._get_moe_dp_group()
        dp_size = dp_group.size() if dp_group is not None else 1
        if dp_group is not None and dist.is_available() and dist.is_initialized():
            num_frames_global = num_frames.clone()
            dist.all_reduce(num_frames_global, op=dist.ReduceOp.SUM, group=dp_group)
        else:
            num_frames_global = num_frames
        num_frames_global = num_frames_global.clamp(min=1)

        if self._fused_ce is not None:
            loss_sum = self._fused_loss_sum(forward_outputs["hidden_states"], inputs["target_ids"])
            loss = loss_sum * dp_size / num_frames_global
        else:
            with loss_parallel():
                logits = forward_outputs["logits"]
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),  # BSHD (B,T,V) or THD (1,T,V) -> (*, V)
                    inputs["target_ids"].reshape(-1),  # BSHD (B,T) or THD (T,) -> (*,)
                    reduction="sum",
                    ignore_index=-100,
                )
                loss = loss_sum * dp_size / num_frames_global

        if (dummy_audio_loss := inputs.get("dummy_audio_loss")) is not None:
            loss = loss + dummy_audio_loss

        # Latent speaker supervision loss (auxiliary, optional). Not available when the fused
        # CE path is active (see _setup_fused_cross_entropy): that path never materializes
        # ``logits``, so lss_loss forces fused CE off at setup time and ``logits`` is always
        # defined here.
        if self.lss_loss is not None and num_frames > 0:
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            loss = loss + self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])

        # Display the local per-token CE so logged values stay on the same scale as before
        # this fix. The gradient-carrying ``loss`` above is the globally-normalized quantity.
        with torch.no_grad():
            loss_display = loss_sum.detach() / num_frames.clamp(min=1)

        # Multi-Token Prediction auxiliary loss. Compute the aggregate and per-head losses
        # in one pass so WandB can show each MTP depth without repeating the expensive
        # lm_head + CE work. ``mtp_loss`` keeps the same meaning as before: the weighted
        # auxiliary loss added to the training objective after the DP-size correction.
        mtp_metrics = {}
        mtp_h = forward_outputs.get("mtp_per_depth_h", None)
        if mtp_h is not None:
            # Under packed THD multiple utterances share one token stream, so the
            # per-depth label roll must not predict the next sequence's first token
            # from the current sequence's last token. Pass cu_seqlens (empty/None for
            # BSHD, where each row is already a single sequence) so the loss derives
            # seq_idx and masks cross-sequence targets.
            mtp_per_depth_targets = inputs.get("mtp_per_depth_targets")
            mtp_cu_seqlens = (
                None if mtp_per_depth_targets is not None else inputs.get("llm_kwargs", {}).get("cu_seqlens")
            )
            with loss_parallel():
                mtp_loss_output = calculate_mtp_loss_with_per_depth(
                    self._mtp_loss_fn,
                    mtp_per_depth_targets=mtp_per_depth_targets,
                    mtp_per_depth_h=mtp_h,
                    labels=inputs["target_ids"],
                    model=self.llm,
                    scaling_factor=self._mtp_loss_scaling_factor,
                    num_label_tokens=num_frames_global,
                    grad_reduce_group=dp_group,
                    cu_seqlens=mtp_cu_seqlens,
                    return_per_depth=True,
                )
                mtp_loss = mtp_loss_output.loss
                mtp_raw_loss_by_head = mtp_loss_output.per_depth_losses
            mtp_loss = dp_size * mtp_loss
            mtp_raw_loss_by_head = [dp_size * head_loss for head_loss in mtp_raw_loss_by_head]
            loss = loss + mtp_loss
            mtp_metrics["mtp_loss"] = mtp_loss.detach()
            for head_idx, head_loss in enumerate(mtp_raw_loss_by_head, start=1):
                mtp_metrics[f"mtp_loss_unscaled/head_{head_idx}"] = head_loss.detach()

        # Input embeds shape is (B, T, H) for BSHD or (T, H) for THD packed.
        input_embeds = inputs["input_embeds"]
        if input_embeds.dim() == 2:
            B, T = 1, input_embeds.shape[0]
        else:
            B, T = input_embeds.shape[:2]
        self._profile_step_memory(B, T)
        if self._bucket_memory and batch_idx > 0 and batch_idx % int(self.cfg.get("bucket_memory_every", 500)) == 0:
            self._dump_bucket_memory()
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "num_frames_global": num_frames_global.to(torch.float32),
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        # batch_size kwarg is required by Lightning when training_step uses
        # the ``dataloader_iter`` signature (it can't auto-infer otherwise).
        self.log("loss", loss_display, on_step=True, prog_bar=True, batch_size=B)
        if mtp_metrics:
            self.log("mtp_loss", mtp_metrics.pop("mtp_loss"), on_step=True, prog_bar=True, batch_size=B)
            self.log_dict(mtp_metrics, on_step=True, batch_size=B)
        self.log_dict({k: v for k, v in ans.items() if k != "loss"}, on_step=True, batch_size=B)
        self.maybe_log_moe_metrics(batch_idx)
        return ans

    def _record_training_stats(self, batch: dict, inputs: dict) -> None:
        # Counters consumed by TrainingStatsCallback. In BSHD, the attention mask
        # counts every real LLM input position. In THD, packed input metadata must
        # come from pre-CP sequence lengths so CP/TP-local tensor shapes do not
        # over- or under-count the global batch.
        if inputs["attention_mask"] is not None:
            num_tokens = inputs["attention_mask"].long().sum()
        else:
            num_tokens = inputs["num_tokens"]
        num_examples = inputs.get("num_examples", batch["input_ids"].shape[0])
        if torch.is_tensor(num_tokens):
            num_tokens = num_tokens.detach().cpu().item()
        if torch.is_tensor(num_examples):
            num_examples = num_examples.detach().cpu().item()
        self._last_batch_num_tokens = int(num_tokens)
        self._last_batch_num_examples = int(num_examples)

    def on_validation_epoch_start(self) -> None:
        self._partial_val_loss_sums = defaultdict(list)
        self._partial_val_corrects = defaultdict(list)
        self._partial_val_num_frames = defaultdict(list)
        self._partial_val_lss = defaultdict(list)
        self._partial_val_mtp_correct = defaultdict(list)
        self._partial_val_mtp_valid = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        accuracies = []
        reduction_group = self._get_moe_dp_group()
        for name, vals in self._partial_val_loss_sums.items():
            loss_sum = torch.stack(vals).sum()
            correct = torch.stack(self._partial_val_corrects[name]).sum().to(loss_sum.dtype)
            num_frames = torch.stack(self._partial_val_num_frames[name]).sum().to(loss_sum.dtype)
            metric_sums = self._reduce_validation_metric_sums(
                torch.stack([loss_sum, correct, num_frames]), reduction_group
            )
            num_frames = metric_sums[2].clamp(min=1)
            val_loss = metric_sums[0] / num_frames
            val_acc = metric_sums[1] / num_frames

            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)

            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)

        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        if getattr(self, "lss_loss", None) is not None:
            lss_vals = []
            for name, vals in self._partial_val_lss.items():
                val_lss = torch.stack(vals).mean()
                self.log(f"val_lss_{name}", val_lss, on_epoch=True, sync_dist=True)
                lss_vals.append(val_lss)
            if lss_vals:
                self.log("val_lss", torch.stack(lss_vals).mean(), on_epoch=True, sync_dist=True)

        # Multi-Token Prediction teacher-forced agreement metrics. Each accumulated per-head
        # count is already a prefix count: depth k contributes only when every draft through
        # depth k agrees with verifier logits from the ground-truth-conditioned validation
        # forward. This is a cheap quality proxy, not speculative-decoding acceptance: exact
        # acceptance requires verifier forwards conditioned on each proposed draft prefix.
        if self._partial_val_mtp_correct:
            agreement_lengths = []
            for name in self._partial_val_mtp_correct:
                per_head, agreement_length = compute_mtp_agreement_lengths(
                    self._partial_val_mtp_correct[name],
                    self._partial_val_mtp_valid[name],
                    reduce_sums=lambda values: self._reduce_validation_metric_sums(values, reduction_group),
                )
                for head_idx, p in enumerate(per_head, start=1):
                    self.log(
                        f"val_mtp_teacher_forced_agreement_{name}/head_{head_idx}",
                        p,
                        on_epoch=True,
                        sync_dist=True,
                    )
                self.log(
                    f"val_mtp_teacher_forced_prefix_length_{name}", agreement_length, on_epoch=True, sync_dist=True
                )
                agreement_lengths.append(agreement_length)
            if agreement_lengths:
                self.log(
                    "val_mtp_teacher_forced_prefix_length",
                    torch.stack(agreement_lengths).mean(),
                    on_epoch=True,
                    sync_dist=True,
                )

        self._partial_val_loss_sums.clear()
        self._partial_val_corrects.clear()
        self._partial_val_num_frames.clear()
        self._partial_val_lss.clear()
        self._partial_val_mtp_correct.clear()
        self._partial_val_mtp_valid.clear()

    def _reduce_validation_metric_sums(self, metric_sums: Tensor, group) -> Tensor:
        if group is not None and dist.is_available() and dist.is_initialized():
            metric_sums = metric_sums.clone()
            dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM, group=group)
        return metric_sums

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            mtp_metrics_disabled_for_cp = self._mtp_enabled and self._context_parallel_size > 1
            inputs = self.prepare_inputs(
                dataset_batch,
                include_mtp_inputs=not mtp_metrics_disabled_for_cp,
            )
            if mtp_metrics_disabled_for_cp:
                logging.warning(
                    "MTP teacher-forced agreement metrics are disabled under context parallelism because "
                    "rank-local verifier predictions cannot be shifted across CP boundaries.",
                    mode=logging_mode.ONCE,
                )
            # Enable MTP only around the validation forward while keeping the model in eval mode.
            with mtp_validation_forward(self.llm, enabled=self._mtp_enabled and not mtp_metrics_disabled_for_cp):
                forward_outputs = self(
                    inputs["input_embeds"],
                    attention_mask=inputs["attention_mask"],
                    **inputs.get("llm_kwargs", {}),
                )
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                logits = forward_outputs["logits"]
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    inputs["target_ids"].reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

            if self.lss_loss is not None and num_frames > 0:
                lss_logits = logits.full_tensor() if isinstance(logits, DTensor) else logits
                log_probs = torch.nn.functional.log_softmax(lss_logits.float(), dim=-1)
                lss_val = self.lss_loss(log_probs=log_probs, labels=inputs["target_ids"])
                self._partial_val_lss[name].append(lss_val.detach())

            verifier_predictions = vocab_parallel_argmax(logits)
            preds = verifier_predictions.view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            correct = preds.eq(refs).sum()

            self._partial_val_loss_sums[name].append(loss_sum.detach())
            self._partial_val_corrects[name].append(correct.detach().to(loss_sum.dtype))
            self._partial_val_num_frames[name].append(num_frames.detach().to(loss_sum.dtype))

            # Multi-Token Prediction teacher-forced prefix agreement for each depth.
            mtp_h = forward_outputs.get("mtp_per_depth_h", None)
            if mtp_h is not None and not mtp_metrics_disabled_for_cp:
                mtp_cu_seqlens = inputs.get("llm_kwargs", {}).get("cu_seqlens")
                correct_by_head, valid_by_head = calculate_mtp_teacher_forced_agreement(
                    mtp_per_depth_h=mtp_h,
                    labels=inputs["target_ids"],
                    model=self.llm,
                    verifier_predictions=verifier_predictions,
                    cu_seqlens=mtp_cu_seqlens,
                )
                self._partial_val_mtp_correct[name].append(torch.stack(correct_by_head).detach().to(torch.int64))
                self._partial_val_mtp_valid[name].append(torch.stack(valid_by_head).detach().to(torch.int64))

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self._setup_moe_fsdp_sync()
        with loss_parallel():
            super().backward(*args, **kwargs)

    def _setup_moe_fsdp_sync(self):
        """Configure MoE FSDP gradient sync for gradient accumulation.

        When ``accumulate_grad_batches > 1``, disables gradient all-reduce and
        resharding on intermediate backward passes and re-enables them on the
        final backward before ``optimizer.step()``.  This avoids redundant
        communication during gradient accumulation.

        Delegates to the LLM's ``MoEFSDPSyncMixin`` methods.  No-op when the
        LLM lacks the mixin or gradient accumulation is not active.
        """
        if not self._use_fsdp or not hasattr(self.llm, 'prepare_for_grad_accumulation'):
            return
        acc = self.trainer.accumulate_grad_batches if self._trainer else 1
        if acc <= 1:
            return
        batch_idx = getattr(self, '_current_batch_idx', 0)
        is_final = (batch_idx + 1) % acc == 0 or (batch_idx + 1) == self.trainer.num_training_batches
        if is_final:
            self.llm.prepare_for_final_backward()
        else:
            self.llm.prepare_for_grad_accumulation()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """Override Lightning's gradient clipping to handle mixed FSDP device meshes.

        When automodel parallelizes the LLM, some parameters end up as DTensors
        on the ``(dp_replicate, dp_shard_cp)`` mesh while others may be on the
        flattened ``dp`` mesh.  PyTorch's ``clip_grad_norm_`` requires all norms
        to share the same mesh for ``torch.stack``.  We delegate to automodel's
        mesh-aware ``_clip_grad_norm_impl`` which groups parameters by
        ``(mesh_id, placements)`` and combines per-group norms as plain tensors.
        """
        if not self._use_fsdp or gradient_clip_val is None or gradient_clip_val <= 0:
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        from nemo_automodel.components.training.utils import _clip_grad_norm_impl

        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if params:
            _clip_grad_norm_impl(params, max_norm=gradient_clip_val)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        spk_targets: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        enable_thinking: bool | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            spk_targets: Optional ``(B, T, n_spk)`` speaker-activity tensor (e.g. oracle / RTTM-derived
                diarization) injected into the perception encoder. Only effective when the mounted
                encoder is a ``ParallelExpertEncoder`` (i.e. ``model.pe_encoder_path`` was set); it
                overrides the encoder's embedded Sortformer prediction for this call. When ``None``
                (default), the encoder runs its embedded Sortformer as usual.
            generation_config: Optional HuggingFace GenerationConfig object.
            enable_thinking: Optional prompt-formatter hint forwarded to ``encode_dialog``.
                Relevant for prompt formats that support thinking/reasoning mode.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts.to(self.device)
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            formatter_kwargs = {}
            if enable_thinking is not None:
                formatter_kwargs["enable_thinking"] = enable_thinking
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt, **formatter_kwargs)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self._embed_tokens(tokens_to_embed)
            if self._uses_parallel_expert_encoder() and spk_targets is None:
                # This is only used for inference when ``spk_targets`` is None.
                # PEE needs to produce ``spk_targets`` itself through recursive encoding.
                self._warn_parallel_expert_encoder_inference_compatibility(cp_size=1)
                audio_embeds, audio_embed_lens = self.perception(input_signal=audios, input_signal_length=audio_lens)
                audio_embeds = [emb[:emblen] for emb, emblen in zip(audio_embeds, audio_embed_lens)]
            else:
                audio_embeds = encode_audio_with_optional_chunking(
                    self.perception,
                    audios,
                    audio_lens,
                    chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                    sampling_rate=self.sampling_rate,
                    spk_targets=spk_targets,
                )
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_id=self.audio_locator_tag_id,
                replacements=audio_embeds,
                target_ids=None,
            )
            answer_tokens = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        else:
            # Text-only generation — embed_tokens stays in LLM, HF generate uses it natively.
            attention_mask = tokens != self.text_pad_id
            answer_tokens = self.llm.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def setup_moe_options(self):
        """Apply MoE config overrides and enable load balance tracking.

        Must be called after ``self.llm`` is created.  Iterates over all Gate
        modules in the LLM and overrides their settings.  Also enables
        load balance tracking when ``moe_metrics.enabled`` is set.

        Safe no-op when the LLM has no Gate modules (non-MoE backbone).
        """
        from nemo_automodel.components.moe.layers import Gate

        aux_loss_coeff = self.cfg.get("aux_loss_coeff", 0.0)
        if aux_loss_coeff > 0:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.aux_loss_coeff = aux_loss_coeff

        train_gate = self.cfg.get("train_gate", False)
        if train_gate:
            for module in self.llm.modules():
                if isinstance(module, Gate):
                    module.train_gate = True
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is not None and moe_metrics_cfg.get("enabled", False):
            from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

            enable_load_balance_tracking(self.llm)

    def maybe_log_moe_metrics(self, step: int):
        """Collect and log MoE load balance metrics.

        All ranks must call this method (the all-reduce inside
        ``collect_expert_loads`` is collective).  Metrics are logged via
        Lightning's ``self.log_dict`` which respects ``log_every_n_steps``.

        Args:
            step: Current ``batch_idx``, used to decide brief vs detailed mode.
        """
        moe_metrics_cfg = self.cfg.get("moe_metrics", None)
        if moe_metrics_cfg is None or not moe_metrics_cfg.get("enabled", False):
            return

        from nemo_automodel.components.moe.load_balance_metrics import (
            collect_expert_loads,
            compute_brief_metrics,
            compute_detailed_metrics,
        )

        dp_group = self._get_moe_dp_group()
        layer_loads = collect_expert_loads(self.llm, dp_group=dp_group)
        if not layer_loads:
            return

        mode = moe_metrics_cfg.get("mode", "brief")
        top_k = moe_metrics_cfg.get("top_k_experts", 5)

        if mode == "detailed":
            detailed_every = moe_metrics_cfg.get("detailed_every_steps", None)
            if detailed_every is not None and step % detailed_every != 0:
                metrics = compute_brief_metrics(layer_loads, top_k=top_k)
            else:
                metrics = compute_detailed_metrics(layer_loads, top_k=top_k)
        else:
            metrics = compute_brief_metrics(layer_loads, top_k=top_k)

        # ``batch_size=1`` is required when training_step uses the
        # ``dataloader_iter`` flavor: Lightning cannot infer the batch size
        # from the closure, and these MoE metrics are model-internal
        # aggregates (load fractions, top-k expert utilization), so the
        # per-call batch_size is just a logging-aggregation hint, not a true
        # sample count. Without it Lightning raises
        # ``MisconfigurationException`` on the very first training step.
        self.log_dict(metrics, on_step=True, batch_size=1)

    def _get_moe_dp_group(self):
        """Return the DP process group for MoE metrics all-reduce.

        Mirrors Automodel's ``_get_dp_group(include_cp=True)`` pattern: prefers
        the ``dp_cp`` submesh (includes context parallelism) for the broadest
        reduction, falling back to ``dp``. ``dp`` and ``dp_cp`` are flattened
        submeshes registered in ``device_mesh._flatten_mapping`` — they are not
        in ``mesh_dim_names`` of the root mesh, so resolve them via
        ``get_flat_mesh`` (same helper Automodel's ``base_recipe`` uses).

        Returns ``None`` when no device mesh is available (e.g. DDP training),
        causing ``collect_expert_loads`` to skip all-reduce (rank-local view).
        """
        device_mesh = getattr(self, "_device_mesh", None)
        if device_mesh is None:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        try:
            if "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
                return get_flat_mesh(device_mesh, "dp_cp").get_group()
            return get_flat_mesh(device_mesh, "dp").get_group()
        except KeyError:
            return None

    def _configure_moe_aux_loss_scaler(self) -> None:
        """Cancel FSDP's gradient averaging on MoE aux-loss grads.

        ``MoEAuxLossAutoScaler`` multiplies aux-loss-derived gradients by
        ``main_loss_backward_scale`` during backward. FSDP's all-reduce then
        divides every gradient by ``dp_group_size``. Setting the scaler to
        ``dp_group_size`` (non-PP case) cancels that division out, matching the
        intent in ``nemo_automodel/recipes/llm/train_ft.py`` — otherwise the
        aux-loss contribution to the gradient would be under-scaled by a factor
        of ``dp_group_size``.

        No-op when ``nemo_automodel`` isn't available (non-MoE builds).
        """
        try:
            from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
        except ImportError:
            return
        dp_group = self._get_moe_dp_group()
        dp_size = dp_group.size() if dp_group is not None else 1
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(dp_size))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def _apply_mtp_training_mode(self, training_mode: str) -> None:
        """Apply the optimizer-facing parameter policy for an MTP run.

        ``joint`` preserves the recipe's ordinary freeze policy while making
        its documented ``llm.mtp`` keep rule wrapper-aware. ``head_only``
        freezes every parameter outside ``llm.mtp`` and guarantees that the
        head wins over any user-supplied ``freeze_params`` expression. The
        latter is intentionally strict: speech encoder, modality adapter,
        backbone, embeddings, and LM head all remain fixed.

        This runs after model/checkpoint setup so loading is unaffected, and
        before Lightning constructs the optimizer.
        """
        if training_mode not in {"joint", "head_only"}:
            return
        if not self._mtp_enabled:
            if training_mode == "head_only":
                raise RuntimeError("MTP training_mode='head_only' requires an attached MTP head.")
            return

        # freeze_and_subset applies recipe regexes when configure_optimizers is
        # called. Resolve the live module path so wrappers such as torch.compile
        # (``_orig_mod``), DDP, or PEFT cannot cause a broad ``^llm\\..+$`` rule
        # to remove the one parameter namespace that head-only mode promises to train.
        mtp_module = self.llm.mtp
        mtp_module_name = next((name for name, module in self.named_modules() if module is mtp_module), None)
        if not mtp_module_name:
            raise RuntimeError("Could not resolve the attached MTP module's parameter namespace.")
        keep_pattern = rf"^{re.escape(mtp_module_name)}\..+$"
        if "prevent_freeze_params" not in self.cfg:
            self.cfg.prevent_freeze_params = []

        if training_mode == "joint":
            canonical_keep_pattern = r"^llm\.mtp\..+$"
            if canonical_keep_pattern in self.cfg.prevent_freeze_params:
                if keep_pattern not in self.cfg.prevent_freeze_params:
                    self.cfg.prevent_freeze_params.append(keep_pattern)
            return

        mtp_param_ids = {id(param) for param in mtp_module.parameters()}
        if not mtp_param_ids:
            raise RuntimeError("MTP training_mode='head_only' found an MTP module with no parameters.")
        for param in self.parameters():
            param.requires_grad_(id(param) in mtp_param_ids)

        if keep_pattern not in self.cfg.prevent_freeze_params:
            self.cfg.prevent_freeze_params.append(keep_pattern)

        trainable = sum(param.numel() for param in self.llm.mtp.parameters() if param.requires_grad)
        logging.info(f"MTP training mode=head_only: trainable MTP parameters={trainable}; all others frozen")

    def configure_model(
        self,
        distributed_setup=None,
        activation_checkpointing_perception: bool | None = None,
    ) -> None:
        if distributed_setup is None and self._trainer is not None:
            distributed_setup = getattr(self._trainer.strategy, "distributed_setup", None)
        if distributed_setup is None:
            distributed_setup = getattr(self, "_distributed_setup", None)

        device_mesh = None
        if distributed_setup is not None:
            self._distributed_setup = distributed_setup
            device_mesh = distributed_setup.mesh_context.device_mesh
            self._device_mesh = device_mesh
            self._moe_mesh = distributed_setup.mesh_context.moe_mesh
        else:
            device_mesh = getattr(self, "_device_mesh", None)

        # Derive dtype from trainer precision (e.g. "bf16-flash" -> bfloat16).
        dtype = torch.float32
        if self._trainer is not None:
            precision = str(self._trainer.precision)
            if "bf16" in precision:
                dtype = torch.bfloat16
            elif "16" in precision:
                dtype = torch.float16
        elif hasattr(self.cfg, 'torch_dtype') and self.cfg.torch_dtype is not None:
            td = self.cfg.torch_dtype
            dtype = getattr(torch, td) if isinstance(td, str) else td

        if activation_checkpointing_perception is None and self._trainer is not None:
            activation_checkpointing_perception = getattr(
                self._trainer.strategy, "activation_checkpointing_perception", None
            )
        if activation_checkpointing_perception is None:
            activation_checkpointing_perception = False
        if distributed_setup is not None and distributed_setup.mesh_context.pp_size > 1:
            raise NotImplementedError("SALMAutomodel does not support pipeline parallelism yet.")

        automodel_kwargs = {}
        if distributed_setup is not None:
            automodel_kwargs["distributed_setup"] = distributed_setup

        # When LoRA is configured and we have a device_mesh, pass peft_config
        # through automodel so LoRA is applied before FSDP2 sharding (handles
        # meta-device init correctly).
        peft_config = make_peft_config(self.cfg.lora) if "lora" in self.cfg else None
        if peft_config is not None and device_mesh is not None:
            automodel_kwargs["peft_config"] = peft_config

        # Pass compile_config through to automodel for torch.compile support.
        compile_cfg = self.cfg.get("compile", None)
        if compile_cfg is not None:
            from nemo_automodel.components.utils.compile_utils import CompileConfig

            compile_dict = dict(compile_cfg)
            automodel_kwargs["compile_config"] = CompileConfig(**compile_dict)

        pretrained_weights = self.cfg.get("pretrained_weights", True)
        pretrained_llm_weights = self.cfg.get("pretrained_llm_weights", pretrained_weights)
        pretrained_asr_weights = self.cfg.get("pretrained_asr_weights", pretrained_weights)
        # Pass backend through to automodel — lets YAML pick attn/linear/rms_norm/MoE
        # dispatcher backends (e.g. set attn=sdpa to bypass TransformerEngine).
        backend_cfg = self.cfg.get("automodel_backend", None)
        if backend_cfg is not None:
            from nemo_automodel.components.models.common import BackendConfig

            automodel_kwargs["backend"] = BackendConfig(**OmegaConf.to_container(backend_cfg, resolve=True))

        # Pin the SDPA kernel used by attn=sdpa (e.g. [flash_attention] to force FA2
        # and error out if unavailable). Accepts strings; resolved by automodel.
        sdpa_method = self.cfg.get("sdpa_method", None)
        if sdpa_method is not None:
            automodel_kwargs["sdpa_method"] = list(OmegaConf.to_container(sdpa_method, resolve=True))

        # Multi-Token Prediction (MTP): load the checkpoint config first and add a
        # missing/replacement head definition before model construction. Automodel can
        # then initialize, EP/FSDP-shard, activation-checkpoint, and compile the MTP
        # sublayers together with the backbone. Native checkpoint MTP config is kept by
        # default; set replace_existing_head=true to use the recipe's head definition.
        mtp_cfg = self.cfg.get("mtp", None)
        mtp_requested = mtp_cfg is not None and mtp_cfg.get("enabled", False)
        mtp_training_mode = str(mtp_cfg.get("training_mode", "joint")) if mtp_requested else "disabled"
        if mtp_requested and mtp_training_mode not in {"joint", "head_only"}:
            raise ValueError(
                f"Unknown mtp.training_mode {mtp_training_mode!r}; expected 'joint' or 'head_only' when MTP is enabled"
            )
        logging.info(f"MTP training mode={mtp_training_mode}")
        if mtp_requested:
            # MTP supports both BSHD and packed THD. For THD the MTP loss must
            # receive cu_seqlens so target rolling is masked at packed sequence
            # boundaries (see training_step); the MTP sublayers already get the
            # THD context (qkv_format/cu_seqlens/seq_idx) from the model forward.
            self._mtp_loss_scaling_factor = float(mtp_cfg.get("loss_scaling_factor", 0.1))
            self._mtp_loss_fn = build_mtp_loss_fn()
            requested_depth = int(mtp_cfg.get("num_nextn_predict_layers", 1))
            use_repeated_layer = bool(mtp_cfg.get("use_repeated_layer", False))
            # HF/vLLM exports describe physical layers. A repeated MTP head has one
            # physical layer even when it performs multiple logical prediction steps.
            physical_depth = 1 if use_repeated_layer else requested_depth
            automodel_kwargs["mtp_config_overrides"] = {
                "num_nextn_predict_layers": physical_depth,
                "mtp_hybrid_override_pattern": str(mtp_cfg.get("hybrid_override_pattern", "*")),
                "mtp_layers_block_type": None,
            }
            automodel_kwargs["replace_mtp_config"] = bool(mtp_cfg.get("replace_existing_head", False))
            automodel_kwargs["mtp_loss_scaling_factor"] = self._mtp_loss_scaling_factor
            if use_repeated_layer:
                # HF exports contain the physical depth so their state dict has the
                # same number of layers on reload. Restore the logical iteration count
                # from the SpeechLM config when constructing that physical head.
                automodel_kwargs["num_nextn_predict_layers"] = requested_depth
                automodel_kwargs["mtp_use_repeated_layer"] = True
        else:
            # Some checkpoints (including Nemotron-3.5 Lightning) ship a native
            # MTP head in config.json. Explicitly override its depth to zero so
            # users who omit the block or set ``mtp.enabled: false`` do not pay
            # the MTP memory/compute cost during SpeechLM fine-tuning.
            automodel_kwargs["num_nextn_predict_layers"] = 0

        self.llm = load_pretrained_automodel_llm(
            self.cfg.pretrained_llm,
            pretrained_weights=pretrained_llm_weights,
            dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            # force_hf: use the HF (bundled remote-code) model, not automodel's
            # native custom impl. Required for dense Nemotron-H (Luciole): automodel
            # registers a native NemotronHForCausalLM for the *v3 MoE* variant and
            # its compat gate only checks `n_routed_experts is not None` — but the
            # dense config carries a stray default n_routed_experts=8, so the gate
            # wrongly passes and the v3 model (no `.backbone`) is built, crashing the
            # parallelizer at `model.backbone.layers`. Forcing HF gives the bundled
            # dense model whose `.backbone` matches the parallelizer / find_embedding_layer.
            force_hf=self.cfg.get("force_hf", False),
            config_overrides=_resolve_llm_config_overrides(self.cfg),
            **automodel_kwargs,
        )
        if mtp_requested and use_repeated_layer:
            # Automodel consumes constructor kwargs that match HF config fields,
            # so the logical iteration count above temporarily overwrites the
            # serialized depth. The built MTPConfig retains that logical count;
            # restore the HF config to the one physical layer saved in the state dict.
            self.llm.config.num_nextn_predict_layers = physical_depth
        if not mtp_requested:
            # The constructor override suppresses a checkpoint-native MTP module but does
            # not mutate the HF config. Keep the serialized config consistent with the
            # actual state dict so conversion/reload does not recreate a missing head.
            self.llm.config.num_nextn_predict_layers = 0

        if mtp_requested and not self._mtp_enabled:
            raise RuntimeError("MTP is enabled but Automodel did not construct an MTP head from the configured model.")
        if not mtp_requested and self._mtp_enabled:
            raise RuntimeError("MTP is disabled but the loaded LLM still has an MTP head attached.")

        # Apply MoE options (aux_loss_coeff override, load balance tracking)
        self.setup_moe_options()

        # Create perception module (must happen after LLM so output_dim matches)
        setup_speech_encoder(self, pretrained_weights=pretrained_asr_weights)

        # Fix projection dim for pretrained_weights=False (config output_dim may not match LLM)
        update_perception_output_dim(self)

        # Activation checkpointing on perception encoder layers. Must run BEFORE
        # FSDP2 wrapping (see LLM path in automodel) so checkpoint_wrapper sees
        # the pristine layer objects and fully_shard indexes the final structure.
        self.perception.set_activation_checkpointing(activation_checkpointing_perception)

        # Apply LoRA adapters to the LLM.
        # When device_mesh is set, LoRA was already applied inside automodel's
        # from_pretrained (before sharding).  Otherwise, apply it now.
        if peft_config is not None and device_mesh is None:
            maybe_install_lora(self)
        elif peft_config is not None:
            # LoRA was applied by automodel; still need to ensure the
            # prevent_freeze_params pattern is set for configure_optimizers.
            ensure_lora_trainable(self)

        if device_mesh is None:
            maybe_load_pretrained_models(self)
            self._apply_mtp_training_mode(mtp_training_mode)
            return

        # Cast perception to training dtype BEFORE FSDP2 wrapping.
        # The LLM is already in the target dtype (loaded via torch_dtype=dtype).
        # FSDP2 requires uniform parameter dtype, so we cast all parameters.
        if dtype != torch.float32:
            self.perception.to(dtype=dtype)

        if device_mesh["tp"].size() > 1:
            self._use_tp = True

        # Use the same FSDP mesh as automodel uses for the LLM so that
        # gradient clipping can torch.stack norms from all parameters.
        dim_names = device_mesh.mesh_dim_names
        if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
        elif "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_shard_cp"]
        else:
            fsdp_mesh = device_mesh["dp"]

        if fsdp_mesh.size() > 1:
            self._use_fsdp = True
            self.perception = fully_shard(self.perception, mesh=fsdp_mesh)

        # Enable MoE FSDP gradient accumulation optimization.
        # The MoEFSDPSyncMixin on the LLM defers gradient sync/resharding on
        # intermediate backward passes — _setup_moe_fsdp_sync() drives it.
        # TODO(pzelasko): causes issue in torch's FSDP backward, investigate later:
        # AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'. Did you mean: 'unsharded_param'?
        # if self._use_fsdp and hasattr(self.llm, 'prepare_for_grad_accumulation'):
        #     self.llm.backend.enable_fsdp_optimizations = True

        # Optionally initialize weights from a previous training checkpoint
        # (fresh optimizer/scheduler). Must happen after FSDP wrapping so that
        # DCP loading can fill DTensor parameters with correct shards.
        maybe_load_pretrained_models(self)

        self._apply_mtp_training_mode(mtp_training_mode)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        embed_tokens = self.embed_tokens
        vocab_size = embed_tokens.num_embeddings if embed_tokens is not None else self.tokenizer.vocab_size
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": vocab_size,
                    "excluded_token_ids": [self.audio_locator_tag_id],
                    "excluded_token_replacement_id": self.text_pad_id,
                    "forced_token_ids": {0: self.audio_locator_tag_id},
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }
