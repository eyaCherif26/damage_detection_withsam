"""Qwen3-VL vision encoder + LoRA + multi-label semantic decoder, registered in MVP_clean as `qwen3vl_semantic`.

Only Qwen3-VL's vision tower is used (27-layer ViT, 1152 channels, 16-px patches for Qwen3-VL-8B); the language
model is never loaded. Only the safetensors shard(s) holding the `model.visual.` weights are downloaded.

Encoder: the image is patchified exactly like Qwen's image processor (each 16x16 patch duplicated over the 2
temporal frames, tokens in 2x2 spatial-merge-block order) and passed to the OFFICIAL Qwen3VLVisionModel.forward,
so position embeddings, 2D rotary and per-image attention are Qwen's own code (works across transformers versions
that provide Qwen3-VL). Hidden states of `feature_layers` are read with forward hooks on the blocks and put back
into raster order -> (B, 1152, H/16, W/16) maps.

Normalisation: MVP_clean feeds ImageNet-normalised images; the backbone converts them back to [0, 1] and applies
Qwen's own mean/std (0.5 / 0.5 for Qwen3-VL-8B, preprocessor_config.json), so MVP_clean stays unchanged.

Tuning modes: "lora" (vision tower frozen, LoRA on `lora_target_modules` + decoder), "frozen" (decoder only),
"full". `backbone_parameters()` returns ONLY the trainable encoder params of the mode, so MVP_clean's
frozen_epochs toggle never unfreezes the base weights.

Decoder `fpn_upernet_lite` + optional hi-res branch: same design as MVP_clean/models/sam_semantic.py (copied here,
since the MVP_clean version on another machine may not contain it).
"""
from __future__ import annotations

import functools
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISUAL_PREFIX = "model.visual."
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # what MVP_clean's transforms normalise with
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SETTINGS = {
    "encoder_tuning": "lora",            # lora | frozen | full
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "lora_dropout": 0.05,
    "lora_target_modules": ["qkv", "proj"],  # Qwen3VLVisionAttention linear layers
    "feature_layers": [8, 16, 24, 26],   # 0-based block indices (deepstack layers + last)
    "hires_branch": True,
    "gradient_checkpointing": True,
    "decoder_dim": 256,
    "image_mean": [0.5, 0.5, 0.5],       # Qwen3-VL preprocessor_config.json
    "image_std": [0.5, 0.5, 0.5],
}


# ---------------------------------------------------------------- weights
def load_qwen_vision_model(model_id: str = DEFAULT_MODEL_ID, cache_dir: str | None = None) -> nn.Module:
    """Qwen3VLVisionModel with pretrained weights; downloads only config.json, the index and the shard(s) that
    contain `model.visual.` keys (one ~5 GB shard for Qwen3-VL-8B, not the whole 17.5 GB model)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vcfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir).vision_config
    vcfg._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(vcfg)
    index = json.loads(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=cache_dir)).read())
    shards = sorted({f for k, f in index["weight_map"].items() if k.startswith(VISUAL_PREFIX)})
    state = {}
    for shard in shards:
        with safe_open(hf_hub_download(model_id, shard, cache_dir=cache_dir), framework="pt") as f:
            for k in f.keys():
                if k.startswith(VISUAL_PREFIX):
                    state[k[len(VISUAL_PREFIX):]] = f.get_tensor(k)
    model.load_state_dict(state, strict=True)
    return model.float()


# ---------------------------------------------------------------- building blocks (as MVP_clean sam_semantic.py)
class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(dropout(x))); base frozen, B zero-initialised (identity at init)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


def _conv_gn(cin: int, cout: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.GroupNorm(min(32, cout // 4) if cout >= 8 else 1, cout),
        nn.GELU(),
    )


class FpnUperNetLite(nn.Module):
    """4 stride-16 ViT feature maps -> pyramid (4/8/16/32) -> FPN -> fused stride-4 map (dim channels)."""

    def __init__(self, in_dim: int, dim: int = 256) -> None:
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2), nn.GroupNorm(32, in_dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(in_dim // 2, in_dim // 2, 2, 2)),  # stride 4
            nn.ConvTranspose2d(in_dim, in_dim, 2, 2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        chans = [in_dim // 2, in_dim, in_dim, in_dim]
        self.lateral = nn.ModuleList([_conv_gn(c, dim, k=1) for c in chans])
        self.smooth = nn.ModuleList([_conv_gn(dim, dim) for _ in chans])
        self.fuse = _conv_gn(4 * dim, dim)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        lat = [l(r(f)) for f, r, l in zip(feats, self.rescale, self.lateral)]
        for i in range(len(lat) - 2, -1, -1):  # top-down
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
        outs = [s(x) for s, x in zip(self.smooth, lat)]
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs[1:]]
        return self.fuse(torch.cat(outs, dim=1))


# ---------------------------------------------------------------- model
def patchify(x: torch.Tensor, patch: int, merge: int, temporal: int) -> torch.Tensor:
    """(B, C, H, W) -> (B * H/patch * W/patch, C * temporal * patch * patch) in Qwen's image-processor layout
    (Qwen2VLImageProcessor.patchify): tokens in merge-block order (block_row, block_col, in_row, in_col), each
    patch as (channel, frame, row, col) with the image repeated over `temporal` frames."""
    b, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    p = x.reshape(b, c, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.permute(0, 2, 5, 3, 6, 1, 4, 7)  # b, bh, bw, ih, iw, c, py, px
    p = p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
    return p.reshape(b * gh * gw, c * temporal * patch * patch)


def tokens_to_map(t: torch.Tensor, b: int, gh: int, gw: int, merge: int) -> torch.Tensor:
    """(B*gh*gw, D) tokens in merge-block order -> (B, D, gh, gw) raster feature map."""
    d = t.shape[-1]
    t = t.reshape(b, gh // merge, gw // merge, merge, merge, d)
    return t.permute(0, 5, 1, 3, 2, 4).reshape(b, d, gh, gw)


class Qwen3VLSemanticModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float, model_id: str = DEFAULT_MODEL_ID,
                 vision_model: nn.Module | None = None, **settings) -> None:
        super().__init__()
        s = {**DEFAULT_SETTINGS, **settings}
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"unknown qwen3vl backbone setting(s) {sorted(unknown)}; known: {sorted(DEFAULT_SETTINGS)}")
        if s["encoder_tuning"] not in ("lora", "frozen", "full"):
            raise ValueError(f"encoder_tuning must be lora/frozen/full, got {s['encoder_tuning']!r}")
        self.settings = s
        self.encoder = vision_model if vision_model is not None else load_qwen_vision_model(model_id)
        cfg = self.encoder.config
        self.patch, self.merge, self.temporal = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
        self.feature_layers = [int(i) for i in s["feature_layers"]]
        if len(self.feature_layers) != 4 or max(self.feature_layers) >= cfg.depth:
            raise ValueError(f"feature_layers must be 4 block indices < depth {cfg.depth}, got {self.feature_layers}")
        hidden = cfg.hidden_size

        if s["encoder_tuning"] in ("lora", "frozen"):
            for p in self.encoder.parameters():
                p.requires_grad = False
        if s["encoder_tuning"] == "lora":
            for blk in self.encoder.blocks:
                for name in s["lora_target_modules"]:
                    base = getattr(blk.attn, name, None)
                    if not isinstance(base, nn.Linear):
                        raise ValueError(f"LoRA target {name!r} is not an nn.Linear in the Qwen3-VL vision attention")
                    setattr(blk.attn, name, LoRALinear(base, s["lora_rank"], s["lora_alpha"], s["lora_dropout"]))
        if s["gradient_checkpointing"]:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # input normalisation: ImageNet (MVP_clean) -> [0,1] -> Qwen mean/std
        self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("q_mean", torch.tensor(s["image_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("q_std", torch.tensor(s["image_std"]).view(1, 3, 1, 1), persistent=False)

        self._captured: dict[int, torch.Tensor] = {}
        for i in sorted(set(self.feature_layers)):  # bound method + partial: picklable (no closure)
            self.encoder.blocks[i].register_forward_hook(functools.partial(self._capture_hook, i))

        dim = s["decoder_dim"]
        self.decoder = FpnUperNetLite(hidden, dim)
        self.hires = nn.Sequential(_conv_gn(3, 32, stride=2), _conv_gn(32, 32)) if s["hires_branch"] else None
        self.head_dropout = nn.Dropout2d(dropout)
        self.refine = _conv_gn(dim + (32 if s["hires_branch"] else 0), 64)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def _capture_hook(self, idx: int, _module, _inputs, output) -> None:
        self._captured[idx] = output[0] if isinstance(output, tuple) else output

    # ---- encoder
    def encode(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = pixel_values.shape
        unit = self.patch * self.merge
        if h % unit or w % unit:
            raise ValueError(f"qwen3vl_semantic needs H and W divisible by {unit}, got {h}x{w}")
        x = (pixel_values * self.in_std + self.in_mean - self.q_mean) / self.q_std
        gh, gw = h // self.patch, w // self.patch
        tokens = patchify(x, self.patch, self.merge, self.temporal)
        grid_thw = torch.tensor([[1, gh, gw]] * b, device=x.device, dtype=torch.long)
        self._captured = {}
        self.encoder(tokens, grid_thw=grid_thw)
        return [tokens_to_map(self._captured[i], b, gh, gw, self.merge) for i in self.feature_layers]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.decoder(self.encode(pixel_values))  # stride 4
        if self.hires is not None:
            hr = self.hires(pixel_values)  # stride 2
            x = torch.cat([F.interpolate(x, size=hr.shape[-2:], mode="bilinear", align_corners=False), hr], dim=1)
        logits = self.classifier(self.refine(self.head_dropout(x)))
        return F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

    # ---- MVP_clean registry contract
    def backbone_parameters(self):
        """Trainable encoder params only: LoRA A/B ("lora"), none ("frozen"), all ("full")."""
        if self.settings["encoder_tuning"] == "lora":
            return [p for n, p in self.encoder.named_parameters() if ".lora_" in n]
        if self.settings["encoder_tuning"] == "frozen":
            return []
        return list(self.encoder.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def trainable_summary(self) -> str:
        trainable = sum(p.numel() for p in self.backbone_parameters()) + sum(p.numel() for p in self.head_parameters())
        total = sum(p.numel() for p in self.parameters())
        return f"trainable {trainable / 1e6:.2f}M / total {total / 1e6:.2f}M ({100 * trainable / total:.1f}%)"


# ---------------------------------------------------------------- MVP_clean registration
def register(settings: dict | None = None, name: str = "qwen3vl_semantic") -> None:
    """Register the backbone in MVP_clean's registry (MVP_clean must be on sys.path). The factory uses
    model.encoder_name (Hugging Face id), model.num_classes and model.dropout from the experiment YAML plus
    `settings` (the backbone YAML)."""
    from models.registry import register_model, registered_models

    if name in registered_models():
        return
    settings = dict(settings or {})

    @register_model(name)
    def build(cfg):
        model_id = cfg.encoder_name or DEFAULT_MODEL_ID
        model = Qwen3VLSemanticModel(num_classes=cfg.num_classes, dropout=cfg.dropout, model_id=model_id, **settings)
        print(f"[{name}] {model_id}, encoder_tuning={model.settings['encoder_tuning']}, {model.trainable_summary()}")
        return model
