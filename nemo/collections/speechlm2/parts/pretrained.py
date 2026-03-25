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
from contextlib import contextmanager
from pathlib import Path

import torch
from omegaconf import open_dict
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule

from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=True)

def find_embedding_layer(llm):
    """
    Locates the embedding layer in the LLM.
    Returns (parent_object, attribute_name) where the embedding is located.
    Returns (None, None) if not found.
    """
    # Paths to try for different model architectures
    paths_to_try = [
        ['backbone', 'embeddings'],      # NemotronH
        ['model', 'embed_tokens'],       # Llama, Mistral, Qwen, standard Nemotron
        ['transformer', 'wte'],          # GPT-2
        ['gpt_neox', 'embed_in'],       # GPT-NeoX
        ['decoder', 'embed_tokens'],     # BART, T5
    ]
    
    # Unwrap PeftModel if present
    if isinstance(llm, PeftModel):
        llm = llm.base_model.model
    
    for path in paths_to_try:
        obj = llm
        try:
            # Navigate to parent
            for attr in path[:-1]:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    break
            else:
                # Successfully navigated to parent
                final_attr = path[-1]
                # Check if this could be the right place
                # We return even if the attribute is missing (it might have been deleted)
                # But we verify the parent exists and usually has this attribute or is the right structure
                return obj, final_attr
        except:
            continue
            
    return None, None

def delete_embeddings(llm):
    """
    Delete embeddings from LLM to save memory.
    Returns True if successful, False otherwise.
    """
    parent, attr_name = find_embedding_layer(llm)
    
    if parent is not None and attr_name is not None:
        if hasattr(parent, attr_name):
            delattr(parent, attr_name)
            return True
            
    return False

@contextmanager
def move_embedding(model):
    """
    Context manager to temporarily restore embeddings to the LLM for generation.
    Handles multiple model architectures automatically.
    """
    parent, attr_name = find_embedding_layer(model.llm)
    
    if parent is None or attr_name is None:
        # Could not find embedding location
        print("⚠ Warning: Could not determine embedding location for move_embedding")
        print(f"  Model type: {type(model.llm).__name__}")
        yield
        return
    
    # Save original state (might be None if deleted)
    original_embed = getattr(parent, attr_name, None)
    
    try:
        # Temporarily restore embeddings for generation
        setattr(parent, attr_name, model.embed_tokens)
        yield
    finally:
        # Restore original state
        if original_embed is not None:
            # Embeddings existed before, restore them
            setattr(parent, attr_name, original_embed)
        else:
            # Embeddings were deleted before, delete them again to save memory
            if hasattr(parent, attr_name):
                delattr(parent, attr_name)
# def move_embedding(model):
#     """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
#     if isinstance(model.llm, PeftModel):
#         model.llm.base_model.model.model.embed_tokens = model.embed_tokens
#     else:
#         model.llm.model.embed_tokens = model.embed_tokens
#     yield
#     if isinstance(model.llm, PeftModel):
#         del model.llm.base_model.model.model.embed_tokens
#     else:
#         del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.
    """
    if pretrained_weights:
        asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
