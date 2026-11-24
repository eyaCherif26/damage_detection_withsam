"""qwen3vl_semantic backbone: CPU tests with a tiny RANDOM Qwen3-VL vision config (no download, no MVP_clean).

Run from the VLM_test folder:  python -m pytest tests/test_qwen3vl_semantic.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_seg.qwen3vl_semantic import Qwen3VLSemanticModel, patchify, tokens_to_map  # noqa: E402

qwen = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
SIZE = 64  # 4x4 patches of 16 px = 2x2 merge blocks


def tiny_vision() -> torch.nn.Module:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    cfg = Qwen3VLVisionConfig(
        depth=3, hidden_size=64, intermediate_size=128, num_heads=4, patch_size=16, spatial_merge_size=2,
        temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=16, deepstack_visual_indexes=[0, 1, 2],
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)  # identical weights on every call
    return qwen.Qwen3VLVisionModel(cfg).eval()


def make(tuning: str = "lora", **kw) -> Qwen3VLSemanticModel:
    settings = dict(encoder_tuning=tuning, feature_layers=[0, 1, 2, 2], gradient_checkpointing=False, lora_rank=4)
    settings.update(kw)
    return Qwen3VLSemanticModel(num_classes=5, dropout=0.0, vision_model=tiny_vision(), **settings)


def test_patchify_matches_qwen_image_processor():
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        ref_fn = Qwen2VLImageProcessor.patchify
    except (ImportError, AttributeError):
        pytest.skip("this transformers version has no Qwen2VLImageProcessor.patchify")
    x = torch.randn(2, 3, SIZE, SIZE)
    ref, gh, gw = ref_fn(None, x, patch_size=16, merge_size=2, temporal_patch_size=2)
    assert (gh, gw) == (SIZE // 16, SIZE // 16)
    torch.testing.assert_close(patchify(x, 16, 2, 2), ref.reshape(-1, ref.shape[-1]))


def test_tokens_to_map_restores_raster_positions():
    vu = pytest.importorskip("transformers.vision_utils")
    gh = gw = 8
    pos = vu.get_vision_position_ids(torch.tensor([[1, gh, gw]]), 2).float()  # (N, 2) row, col in token order
    m = tokens_to_map(pos, 1, gh, gw, 2)
    torch.testing.assert_close(m[0, 0], torch.arange(gh).float()[:, None].expand(gh, gw))
    torch.testing.assert_close(m[0, 1], torch.arange(gw).float()[None, :].expand(gh, gw))


def test_features_match_official_vision_forward():
    model = make("frozen", hires_branch=False).eval()
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        ours = model.encode(x)[-1]  # feature of block 2 = last block
        xq = (x * model.in_std + model.in_mean - model.q_mean) / model.q_std
        out = model.encoder(patchify(xq, 16, 2, 2), grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]))
        last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    torch.testing.assert_close(ours, tokens_to_map(last, 2, 4, 4, 2))


def test_output_shape_and_trainable_set():
    model = make("lora")
    out = model(torch.randn(2, 3, SIZE, SIZE))
    assert out.shape == (2, 5, SIZE, SIZE)
    names = {id(p): n for n, p in model.named_parameters()}
    enc_trainable = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
    assert enc_trainable and all(".lora_A." in n or ".lora_B." in n for n in enc_trainable)
    assert {names[id(p)] for p in model.backbone_parameters()} == {"encoder." + n for n in enc_trainable}
    assert not any(names[id(p)].startswith("encoder.") for p in model.head_parameters())


def test_lora_is_identity_at_init():
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        a = make("frozen").eval().encode(x)
        b = make("lora").eval().encode(x)
    for fa, fb in zip(a, b):
        torch.testing.assert_close(fa, fb)


def test_unfreezing_backbone_never_unfreezes_base_weights():
    model = make("lora")
    for frozen in (True, False):  # mirrors MVP_clean training/trainer.py::_set_backbone_frozen
        for p in model.backbone_parameters():
            p.requires_grad = not frozen
    assert [n for n, p in model.encoder.named_parameters() if ".lora_" not in n and p.requires_grad] == []


def test_frozen_and_full_modes():
    assert make("frozen").backbone_parameters() == []
    full = make("full")
    assert len(full.backbone_parameters()) == len(list(full.encoder.parameters()))


def test_input_size_must_be_multiple_of_32():
    with pytest.raises(ValueError, match="divisible"):
        make("frozen")(torch.randn(1, 3, 48, 48))
