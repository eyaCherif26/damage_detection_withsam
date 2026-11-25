"""Qwen3-VL (vision encoder + language model) semantic-segmentation backbone, registered in MVP_clean as
`qwen3vl_vl_semantic` - first experiment of the "image + text instruction -> Qwen3-VL -> spatial features +
semantic grounding -> segmentation decoder" design, with ONE output: the 5 per-class multi-label masks.

One forward pass of the full Qwen3VLModel (default Qwen/Qwen3-VL-4B-Instruct, bf16) on the image + a fixed text
instruction (Qwen chat format: <|im_start|>user\\n<|vision_start|><|image_pad|>x N<|vision_end|>instruction
<|im_end|>\\n):
  * spatial features: hidden states of the vision blocks in `feature_layers` (default: the model's deepstack
    layers + last block), via forward hooks -> (B, C_vis, H/16, W/16), as in qwen3vl_semantic;
  * semantic grounding: the language model's hidden states (`llm_layers`, default last) at the image-token
    positions -> (B, C_text, H/32, W/32): image regions after the LLM has read them together with the instruction.
The grounding map is projected to the decoder width and added to the decoder's stride-32 level; the rest is the
same decoder / hi-res branch / output as qwen3vl_semantic.

Training: vision encoder frozen + LoRA (or frozen / full, `encoder_tuning`), decoder trained; the language model
is FROZEN and NOT part of this module's parameters or state_dict (kept outside the module tree), so checkpoints
stay small and MVP_clean's optimizer never sees it. Gradients still flow through it into the vision LoRA.
"""
from __future__ import annotations

import functools
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3vl_semantic import (
    DEFAULT_SETTINGS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FpnUperNetLite,
    LoRALinear,
    _conv_gn,
    patchify,
    tokens_to_map,
)

DEFAULT_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
VL_DEFAULT_SETTINGS = {
    **DEFAULT_SETTINGS,
    "feature_layers": None,  # None = deepstack layers + last block of the loaded model ([5, 11, 17, 23] for 4B)
    "instruction": "Segment cracks, spalling, efflorescence, wet spots and rust.",
    "llm_layers": [-1],      # language-model hidden_states indices (0 = embeddings, -1 = last)
}


def load_qwen_vl(model_id: str = DEFAULT_VL_MODEL_ID):
    """Full Qwen3VLModel (no LM head) in bf16 + its tokenizer."""
    from transformers import AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    try:
        vlm = Qwen3VLModel.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    except TypeError:  # older transformers: torch_dtype
        vlm = Qwen3VLModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return vlm.eval(), AutoTokenizer.from_pretrained(model_id)


class Qwen3VLVLSemanticModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float, model_id: str = DEFAULT_VL_MODEL_ID,
                 vlm: nn.Module | None = None, tokenizer=None, **settings) -> None:
        super().__init__()
        s = {**VL_DEFAULT_SETTINGS, **settings}
        unknown = set(settings) - set(VL_DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown qwen3vl_vl backbone setting(s) {sorted(unknown)}; known: {sorted(VL_DEFAULT_SETTINGS)}")
        if s["encoder_tuning"] not in ("lora", "frozen", "full"):
            raise ValueError(f"encoder_tuning must be lora/frozen/full, got {s['encoder_tuning']!r}")
        self.settings = s
        if vlm is None:
            vlm, tokenizer = load_qwen_vl(model_id)
        self._vlm = [vlm]          # plain list: the language model is NOT a registered submodule
        self.tokenizer = tokenizer
        for p in vlm.parameters():
            p.requires_grad = False
        self.visual = vlm.visual   # registered: vision weights + LoRA go into the checkpoint
        vcfg = vlm.config.vision_config
        self.patch, self.merge, self.temporal = vcfg.patch_size, vcfg.spatial_merge_size, vcfg.temporal_patch_size
        fl = s["feature_layers"]
        self.feature_layers = [int(i) for i in (fl if fl is not None else
                                                 list(vcfg.deepstack_visual_indexes) + [vcfg.depth - 1])]
        if len(self.feature_layers) != 4 or max(self.feature_layers) >= vcfg.depth:
            raise ValueError(f"feature_layers must be 4 block indices < depth {vcfg.depth}, got {self.feature_layers}")
        self.llm_layers = [int(i) for i in s["llm_layers"]]
        self.image_token_id = vlm.config.image_token_id

        if s["encoder_tuning"] == "full":
            for p in self.visual.parameters():
                p.requires_grad = True
        if s["encoder_tuning"] == "lora":
            for blk in self.visual.blocks:
                for name in s["lora_target_modules"]:
                    base = getattr(blk.attn, name, None)
                    if not isinstance(base, nn.Linear):
                        raise ValueError(f"LoRA target {name!r} is not an nn.Linear in the Qwen3-VL vision attention")
                    setattr(blk.attn, name, LoRALinear(base, s["lora_rank"], s["lora_alpha"], s["lora_dropout"]))
        if s["gradient_checkpointing"]:
            kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
            self.visual.gradient_checkpointing_enable(**kw)
            vlm.language_model.gradient_checkpointing_enable(**kw)
        self._accepts_mm_types = "mm_token_type_ids" in inspect.signature(vlm.forward).parameters

        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("q_mean", torch.tensor(s["image_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("q_std", torch.tensor(s["image_std"]).view(1, 3, 1, 1), persistent=False)

        self._captured: dict[int, torch.Tensor] = {}
        for i in sorted(set(self.feature_layers)):  # bound method + partial: picklable
            self.visual.blocks[i].register_forward_hook(functools.partial(self._capture_hook, i))
        self._prompt_cache: dict[int, torch.Tensor] = {}

        dim = s["decoder_dim"]
        text_dim = vlm.config.text_config.hidden_size
        self.decoder = FpnUperNetLite(vcfg.hidden_size, dim)
        self.lang_proj = _conv_gn(text_dim * len(self.llm_layers), dim, k=1)
        self.hires = nn.Sequential(_conv_gn(3, 32, stride=2), _conv_gn(32, 32)) if s["hires_branch"] else None
        self.head_dropout = nn.Dropout2d(dropout)
        self.refine = _conv_gn(dim + (32 if s["hires_branch"] else 0), 64)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def _capture_hook(self, idx: int, _module, _inputs, output) -> None:
        self._captured[idx] = output[0] if isinstance(output, tuple) else output

    def train(self, mode: bool = True):
        super().train(mode)
        self._vlm[0].language_model.train(mode)  # not a submodule; needed for its gradient checkpointing
        return self

    def _prompt_ids(self, n_image_tokens: int) -> torch.Tensor:
        """Token ids of one user turn: image placeholder (n tokens) + instruction, Qwen3-VL chat format."""
        if n_image_tokens not in self._prompt_cache:
            text = ("<|im_start|>user\n<|vision_start|>" + "<|image_pad|>" * n_image_tokens + "<|vision_end|>"
                    + self.settings["instruction"] + "<|im_end|>\n")
            ids = torch.as_tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
            if int((ids == self.image_token_id).sum()) != n_image_tokens:
                raise ValueError("tokenizer did not map <|image_pad|> to one image token per placeholder")
            self._prompt_cache[n_image_tokens] = ids
        return self._prompt_cache[n_image_tokens]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vlm = self._vlm[0]
        dev = pixel_values.device
        if vlm.language_model.get_input_embeddings().weight.device != dev:
            vlm.to(dev)  # MVP_clean moves only registered modules; the language model follows here
        b, _, h, w = pixel_values.shape
        unit = self.patch * self.merge
        if h % unit or w % unit:
            raise ValueError(f"qwen3vl_vl_semantic needs H and W divisible by {unit}, got {h}x{w}")
        x = (pixel_values * self.in_std + self.in_mean - self.q_mean) / self.q_std
        gh, gw = h // self.patch, w // self.patch
        n_img = (gh // self.merge) * (gw // self.merge)
        ids = self._prompt_ids(n_img).to(dev).unsqueeze(0).expand(b, -1)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=patchify(x, self.patch, self.merge,
                      self.temporal), image_grid_thw=torch.tensor([[1, gh, gw]] * b, device=dev, dtype=torch.long),
                      output_hidden_states=True, use_cache=False)
        if self._accepts_mm_types:
            kwargs["mm_token_type_ids"] = (ids == self.image_token_id).int()
        self._captured = {}
        out = vlm(**kwargs)

        feats = [tokens_to_map(self._captured[i], b, gh, gw, self.merge).float() for i in self.feature_layers]
        img_pos = ids[0] == self.image_token_id  # same layout for every image of the batch
        lang = torch.cat([out.hidden_states[i][:, img_pos] for i in self.llm_layers], dim=-1).float()
        lang = lang.reshape(b, gh // self.merge, gw // self.merge, -1).permute(0, 3, 1, 2)  # merged tokens: raster

        y = self.decoder(feats, extra=self.lang_proj(lang))  # stride 4
        if self.hires is not None:
            hr = self.hires(pixel_values)
            y = torch.cat([F.interpolate(y, size=hr.shape[-2:], mode="bilinear", align_corners=False), hr], dim=1)
        logits = self.classifier(self.refine(self.head_dropout(y)))
        return F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

    # ---- MVP_clean registry contract
    def backbone_parameters(self):
        if self.settings["encoder_tuning"] == "lora":
            return [p for n, p in self.visual.named_parameters() if ".lora_" in n]
        if self.settings["encoder_tuning"] == "frozen":
            return []
        return list(self.visual.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("visual.")]

    def trainable_summary(self) -> str:
        trainable = sum(p.numel() for p in self.backbone_parameters()) + sum(p.numel() for p in self.head_parameters())
        total = sum(p.numel() for p in self.parameters()) + sum(
            p.numel() for p in self._vlm[0].language_model.parameters())
        return f"trainable {trainable / 1e6:.2f}M / total {total / 1e6:.0f}M ({100 * trainable / total:.2f}%)"


def register(settings: dict | None = None, name: str = "qwen3vl_vl_semantic") -> None:
    """Register in MVP_clean's registry (MVP_clean on sys.path). Uses model.encoder_name (default Qwen3-VL-4B),
    model.num_classes, model.dropout + the backbone YAML settings."""
    from models.registry import register_model, registered_models

    if name in registered_models():
        return
    settings = dict(settings or {})

    @register_model(name)
    def build(cfg):
        model_id = cfg.encoder_name or DEFAULT_VL_MODEL_ID
        model = Qwen3VLVLSemanticModel(num_classes=cfg.num_classes, dropout=cfg.dropout, model_id=model_id, **settings)
        print(f"[{name}] {model_id}, encoder_tuning={model.settings['encoder_tuning']}, "
              f"feature_layers={model.feature_layers}, llm_layers={model.llm_layers}, {model.trainable_summary()}")
        return model
