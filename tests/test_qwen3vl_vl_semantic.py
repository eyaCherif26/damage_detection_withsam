"""qwen3vl_vl_semantic backbone: CPU tests with a tiny RANDOM full Qwen3-VL (vision + language model) and a stub
tokenizer (no download, no MVP_clean).  Run from the VLM_test folder:
    python -m pytest tests/test_qwen3vl_vl_semantic.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_seg.qwen3vl_vl_semantic import Qwen3VLVLSemanticModel  # noqa: E402

qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
SIZE = 64  # 4x4 patches -> 2x2 = 4 image tokens
IMG, VSTART, VEND = 5, 3, 4


class StubTokenizer:
    """Special tokens -> fixed ids (<|image_pad|> -> IMG), every other character -> one text id (vocab 32), so the
    ids follow the real order of image and instruction in the prompt."""

    SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|vision_start|>": VSTART, "<|vision_end|>": VEND,
               "<|image_pad|>": IMG}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [self.SPECIAL[part]] if part in self.SPECIAL else [10 + ord(c) % 22 for c in part]
        return {"input_ids": ids}


def tiny_vlm():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    cfg = Qwen3VLConfig(
        vision_config=dict(depth=3, hidden_size=64, intermediate_size=128, num_heads=4, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32,
                           num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2],
                           rope_parameters={"rope_type": "axial", "rope_theta": 10000.0}),
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
        image_token_id=IMG, video_token_id=6, vision_start_token_id=VSTART, vision_end_token_id=VEND,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return qwen.Qwen3VLModel(cfg).eval()


def make(**kw) -> Qwen3VLVLSemanticModel:
    settings = dict(encoder_tuning="lora", gradient_checkpointing=False, lora_rank=4)
    settings.update(kw)
    return Qwen3VLVLSemanticModel(num_classes=5, dropout=0.0, vlm=tiny_vlm(), tokenizer=StubTokenizer(), **settings)


def test_output_shape():
    model = make()
    assert model.feature_layers == [0, 1, 2, 2]  # deepstack layers + last block, from the config
    assert model(torch.randn(2, 3, SIZE, SIZE)).shape == (2, 5, SIZE, SIZE)


def test_language_model_frozen_and_not_in_module():
    model = make()
    lm = model._vlm[0].language_model
    assert not any(p.requires_grad for p in lm.parameters())
    own = {id(p) for p in model.parameters()}
    assert not any(id(p) in own for p in lm.parameters())
    assert not any(k.startswith("_vlm") or "language_model" in k for k in model.state_dict())


def test_trainable_set_and_gradient_reaches_vision_lora():
    model = make().train()
    model(torch.randn(2, 3, SIZE, SIZE)).sum().backward()
    lora_b = [m.lora_B.weight for blk in model.visual.blocks for m in (blk.attn.qkv, blk.attn.proj)]
    assert any(w.grad is not None and w.grad.abs().sum() > 0 for w in lora_b)
    enc_trainable = {n for n, p in model.visual.named_parameters() if p.requires_grad}
    assert enc_trainable and all(".lora_" in n for n in enc_trainable)


def test_prompt_has_one_image_token_per_merged_patch():
    model = make()
    ids = model._prompt_ids(4)
    assert int((ids == IMG).sum()) == 4


def test_llm_lora_trained_and_saved_without_base_llm():
    model = make(llm_tuning="lora", llm_lora_rank=2).train()
    assert len(model.llm_lora) == 3 * 4  # 3 tiny LLM layers x q/k/v/o
    sd = model.state_dict()
    assert any(k.startswith("llm_lora.") for k in sd)
    assert not any(k.startswith("_vlm") or "language_model" in k for k in sd)  # base LLM still not saved
    llm_ids = {id(p) for p in model.llm_lora.parameters()}
    assert llm_ids <= {id(p) for p in model.backbone_parameters()}
    assert not llm_ids & {id(p) for p in model.head_parameters()}
    model(torch.randn(2, 3, SIZE, SIZE)).sum().backward()
    assert any(m["B"].weight.grad is not None and m["B"].weight.grad.abs().sum() > 0 for m in model.llm_lora)
    lm = model._vlm[0].language_model
    assert not any(p.requires_grad for n, p in lm.named_parameters() if ".lora_" not in n)


@pytest.mark.parametrize("position", ["after", "before"])
def test_instruction_reaches_image_tokens_only_when_before(position):
    """Causal LLM: with the instruction AFTER the image, changing it cannot change the output; BEFORE, it does."""
    model = make(instruction_position=position).eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        model.settings["instruction"], model._prompt_cache = "Segment cracks.", {}
        a = model(x)
        model.settings["instruction"], model._prompt_cache = "Find rust stains and wet spots on piers.", {}
        b = model(x)
    if position == "after":
        assert torch.allclose(a, b, atol=1e-5)
    else:
        assert not torch.allclose(a, b, atol=1e-5)
