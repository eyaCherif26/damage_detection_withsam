"""Per-class pixel evaluation of SAM 3 (zero-shot, or + LoRA weights) on a SAM3_LoRA-layout split.

Metric definitions follow MVP_clean exactly (MVP_clean/metrics/pixel_metrics.py::PixelMetricAccumulator):
per class, the union of all predicted instances (instance score >= threshold, mask prob > 0.5) is compared with
the union of the GT instances; TP/FP/FN are summed over the whole split, then
  IoU = TP/(TP+FP+FN), Dice (= F1) = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN)
plus `macro` (all classes), `macro_present` (classes present in GT) and `frequency_weighted` averages.
Black no-data borders are excluded like MVP_clean (MVP_clean/data/validity_mask.py::no_data_validity_mask).
Differences to MVP_clean, by construction: full image resolution (not 512), fixed threshold (no per-class tuning).

Used two ways:
  * imported by the patched SAM3_LoRA/train_sam3_lora_native.py every epoch (evaluate_split + W&B helpers),
    logging the same W&B keys as MVP_clean (train/loss, val/macro_dice, gap/..., val/per_class_table, ...).
  * CLI, run from the SAM3_LoRA repo root (zero-shot: leave out --lora_weights):
      cd SAM3_LoRA && python ../scripts/sam3_eval.py --split_dir ../data/cif_sam3_full/valid \
          --config ../configs/cif_24gb.yaml --lora_weights ../outputs/sam3_cif_24gb/best_lora_weights.pt \
          --out ../outputs/eval_lora_valid [--wandb]
    Outputs: metrics.json, metrics.csv, predictions.json (COCO results: image_id, file_name, category_id,
    category, prompt, score, bbox [x,y,w,h], area, segmentation RLE), <image>_gt_vs_pred.jpg panels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage
from tqdm import tqdm

EPS = 1e-6
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 255, 0)]  # RGB


# ---------------------------------------------------------------- metrics (MVP_clean definitions)
def no_data_validity(image: np.ndarray, tolerance: int = 3, min_region_px: int = 64) -> np.ndarray:
    """(H, W) bool, True = valid. Same rule as MVP_clean: 8-connected near-black (all channels <= tolerance)
    regions of at least min_region_px pixels are no-data; isolated dark pixels stay valid."""
    candidate = (image <= tolerance).all(axis=-1)
    labeled, n = ndimage.label(candidate, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.ones(candidate.shape, dtype=bool)
    is_no_data = np.bincount(labeled.ravel()) >= min_region_px
    is_no_data[0] = False
    return ~is_no_data[labeled]


class PixelMetrics:
    """numpy twin of MVP_clean's PixelMetricAccumulator (same sums, eps and output structure)."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        n = len(class_names)
        self.tp, self.fp, self.fn, self.gt_pixels = (np.zeros(n, dtype=np.float64) for _ in range(4))

    def update(self, idx: int, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
        pred, gt = pred & valid, gt & valid
        self.tp[idx] += np.count_nonzero(pred & gt)
        self.fp[idx] += np.count_nonzero(pred & ~gt)
        self.fn[idx] += np.count_nonzero(~pred & gt)
        self.gt_pixels[idx] += np.count_nonzero(gt)

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        m = {
            "iou": tp / (tp + fp + fn + EPS),
            "dice": 2 * tp / (2 * tp + fp + fn + EPS),
            "precision": tp / (tp + fp + EPS),
            "recall": tp / (tp + fn + EPS),
        }
        weights = self.gt_pixels / max(self.gt_pixels.sum(), EPS)
        present = self.gt_pixels > 0
        return {
            "per_class": {
                name: {**{k: float(v[i]) for k, v in m.items()}, "gt_pixels": int(self.gt_pixels[i])}
                for i, name in enumerate(self.class_names)
            },
            "macro_present": {k: float(v[present].mean()) if present.any() else float("nan") for k, v in m.items()},
            "macro": {k: float(v.mean()) for k, v in m.items()},
            "frequency_weighted": {k: float((v * weights).sum()) for k, v in m.items()},
        }


# ---------------------------------------------------------------- evaluation loop
def _gt_union(anns: list[dict], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for a in anns:
        seg = a["segmentation"]
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
        m |= mask_utils.decode(rle).astype(bool)
    return m


def _flatten(masks: list[np.ndarray]) -> np.ndarray:
    """Per-class masks -> one label map (class i -> i+1, highest index wins), as MVP_clean's W&B overlay."""
    flat = np.zeros(masks[0].shape, dtype=np.uint8)
    for i, m in enumerate(masks):
        flat[m] = i + 1
    return flat


def evaluate_split(
    model,
    split_dir: str | Path,
    max_images: int | None = None,
    threshold: float = 0.5,
    phrases: dict[str, list[str]] | None = None,
    num_vis: int = 0,
    vis_max_side: int = 512,
    no_data_validity: bool = True,
    collect_predictions: bool = False,
    bf16: bool = False,
    progress: bool = True,
) -> dict:
    """Runs SAM 3 (`model`, already on GPU, in eval mode) with one text query per category (or each phrase of
    `phrases[category name]`, merged) on the first `max_images` images of `split_dir`.

    Returns {"metrics": PixelMetrics.compute(), "class_names": [...], "vis": [(img, gt_flat, pred_flat)],
             "predictions": [COCO result dicts] (only if collect_predictions)}.
    Class names are the categories' `mvp_class` field when present (MVP_clean names), else their prompt name."""
    from sam3.model.sam3_image_processor import Sam3Processor

    split_dir = Path(split_dir)
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_names = [c.get("mvp_class", c["name"]) for c in cats]
    queries = [(phrases or {}).get(c["name"], [c["name"]]) for c in cats]
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    images = coco["images"][:max_images] if max_images else coco["images"]

    proc = Sam3Processor(model, confidence_threshold=threshold)
    acc = PixelMetrics(class_names)
    vis, predictions = [], []
    for k, im in enumerate(tqdm(images, desc=f"eval {split_dir.name}", disable=not progress)):
        img = np.array(Image.open(split_dir / im["file_name"]).convert("RGB"))
        h, w = img.shape[:2]
        valid = no_data_validity_fn(img) if no_data_validity else np.ones((h, w), dtype=bool)
        gts, preds = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            state = proc.set_image(Image.fromarray(img))
            for i, (cat, cat_queries) in enumerate(zip(cats, queries)):
                pred = np.zeros((h, w), dtype=bool)
                for phrase in cat_queries:
                    state = proc.set_text_prompt(prompt=phrase, state=state)
                    masks = state["masks"].squeeze(1).cpu().numpy()
                    if len(masks):
                        pred |= masks.any(0)
                    if collect_predictions:
                        boxes = state["boxes"].float().cpu().tolist()
                        scores = state["scores"].float().cpu().tolist()
                        for m, (x0, y0, x1, y1), s in zip(masks, boxes, scores):
                            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                            rle["counts"] = rle["counts"].decode("ascii")
                            predictions.append({
                                "image_id": im["id"], "file_name": im["file_name"], "category_id": cat["id"],
                                "category": class_names[i], "prompt": phrase, "score": round(s, 4),
                                "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                                "area": int(m.sum()), "segmentation": rle,
                            })
                    proc.reset_all_prompts(state)
                gt = _gt_union([a for a in by_img[im["id"]] if a["category_id"] == cat["id"]], h, w)
                acc.update(i, pred, gt, valid)
                gts.append(gt & valid)
                preds.append(pred & valid)
        if k < num_vis:
            s = min(1.0, vis_max_side / max(h, w))
            size = (max(1, round(w * s)), max(1, round(h * s)))
            vis.append((
                cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                cv2.resize(_flatten(gts), size, interpolation=cv2.INTER_NEAREST),
                cv2.resize(_flatten(preds), size, interpolation=cv2.INTER_NEAREST),
            ))
    return {"metrics": acc.compute(), "class_names": class_names, "vis": vis, "predictions": predictions}


no_data_validity_fn = no_data_validity  # alias: evaluate_split's flag argument shadows the function name


# ---------------------------------------------------------------- heat-map (semantic) evaluation, opt-in
class HeatmapMetrics:
    """Per-class pixel metrics from continuous heat maps, threshold-free: heat values are binned (1/bins), so
    TP/FP/FN at every threshold k/bins, the best-Dice threshold and pixel AP come from two histograms per class.
    Same formulas/eps as PixelMetrics (= MVP_clean)."""

    def __init__(self, class_names: list[str], bins: int = 100) -> None:
        self.class_names, self.bins = class_names, bins
        self.pos = np.zeros((len(class_names), bins), dtype=np.float64)  # GT pixels per heat bin
        self.neg = np.zeros((len(class_names), bins), dtype=np.float64)  # non-GT pixels per heat bin

    def update(self, idx: int, heat: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
        q = np.minimum((heat * self.bins).astype(np.int64), self.bins - 1)
        self.pos[idx] += np.bincount(q[gt & valid], minlength=self.bins)
        self.neg[idx] += np.bincount(q[~gt & valid], minlength=self.bins)

    def _counts(self, i: int, k: int) -> tuple[float, float, float]:
        """TP, FP, FN for class i when predicting heat >= k / bins."""
        return self.pos[i, k:].sum(), self.neg[i, k:].sum(), self.pos[i, :k].sum()

    def compute(self, thresholds: dict | None = None, fixed: float = 0.5) -> dict:
        per = {}
        for i, c in enumerate(self.class_names):
            t = (thresholds or {}).get(c, fixed)
            tp, fp, fn = self._counts(i, int(round(t * self.bins)))
            # sweep: best Dice over thresholds 1/bins .. (bins-1)/bins
            ks = np.arange(1, self.bins)
            tps = np.array([self.pos[i, k:].sum() for k in ks])
            fps = np.array([self.neg[i, k:].sum() for k in ks])
            fns = self.pos[i].sum() - tps
            dices = 2 * tps / (2 * tps + fps + fns + EPS)
            b = int(np.argmax(dices))
            # pixel AP: precision-recall from the highest heat bin downwards (step interpolation)
            ctp, cfp = np.cumsum(self.pos[i, ::-1]), np.cumsum(self.neg[i, ::-1])
            total = self.pos[i].sum()
            rec = ctp / max(total, EPS)
            prec = ctp / np.maximum(ctp + cfp, EPS)
            ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec)) if total > 0 else float("nan")
            per[c] = {"threshold": round(float(t), 3), "iou": tp / (tp + fp + fn + EPS),
                      "dice": 2 * tp / (2 * tp + fp + fn + EPS), "precision": tp / (tp + fp + EPS),
                      "recall": tp / (tp + fn + EPS), "best_threshold": round(float(ks[b] / self.bins), 3),
                      "best_dice": float(dices[b]), "pixel_ap": ap, "gt_pixels": int(total)}
        present = [c for c in self.class_names if per[c]["gt_pixels"] > 0]
        mean = lambda cs, k: float(np.mean([per[c][k] for c in cs])) if cs else float("nan")
        keys = ("iou", "dice", "precision", "recall", "best_dice", "pixel_ap")
        return {"per_class": per,
                "macro": {k: mean(self.class_names, k) for k in keys if k != "pixel_ap"},
                "macro_present": {k: mean(present, k) for k in keys}}


@torch.inference_mode()
def class_heatmap(proc, state: dict, phrase: str, out_hw: tuple[int, int], return_stats: bool = False):
    """Heat map for one text phrase: per pixel the max over ALL of SAM 3's query slots of
    (instance score x mask probability) - no confidence threshold, no NMS. Mirrors
    Sam3Processor._forward_grounding (score = query prob x presence prob), computed at mask resolution and
    upsampled once to out_hw. return_stats=True also returns {presence, max_query_prob, max_score}."""
    import torch.nn.functional as F

    model = proc.model
    state["backbone_out"].update(model.backbone.forward_text([phrase], device=proc.device))
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = model._get_dummy_prompt()
    out = model.forward_grounding(backbone_out=state["backbone_out"], find_input=proc.find_stage,
                                  geometric_prompt=state["geometric_prompt"], find_target=None)
    scores = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
    heat = (scores[:, None, None] * out["pred_masks"][0].sigmoid()).amax(0)  # (h, w)
    heat = F.interpolate(heat[None, None].float(), size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    stats = {"presence": float(out["presence_logit_dec"].sigmoid().max()),
             "max_query_prob": float(out["pred_logits"].sigmoid().max()),
             "max_score": float(scores.max())}
    proc.reset_all_prompts(state)
    heat = heat.clamp(0, 1).cpu().numpy()
    return (heat, stats) if return_stats else heat


def _put_label(panel: np.ndarray, text: str) -> np.ndarray:
    p = panel.copy()
    cv2.putText(p, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(p, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def heat_strip(img: np.ndarray, gts: list, heats: list, class_names: list[str], stats: list[dict],
               side: int = 384) -> np.ndarray:
    """One RGB strip: image + coloured GT, then per class the heat map (JET colormap: blue low .. red high)
    over the image with the GT outline in white, labelled with max heat and SAM 3's presence score."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    base = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gt_panel = base.copy()
    small_gts = [cv2.resize(g.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) for g in gts]
    for i, g in enumerate(small_gts):
        m = g.astype(bool)
        gt_panel[m] = (0.5 * gt_panel[m] + 0.5 * np.array(COLORS[i % len(COLORS)])).astype(np.uint8)
    panels = [_put_label(gt_panel, "image + GT")]
    for name, heat, g, st in zip(class_names, heats, small_gts, stats):
        hs = cv2.resize(heat.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        cm = cv2.applyColorMap(np.round(hs * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        p = np.ascontiguousarray((0.55 * cm + 0.45 * base).astype(np.uint8))
        contours, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p, contours, -1, (255, 255, 255), 1)
        panels.append(_put_label(p, f"{name} max={st['max_heat']:.2f} pres={st['presence']:.2f}"))
    return np.hstack(panels)


def evaluate_split_heatmap(model, split_dir: str | Path, max_images: int | None = None,
                           phrases: dict[str, list[str]] | None = None, no_data_validity: bool = True,
                           bf16: bool = False, save_dir: Path | None = None, progress: bool = True,
                           vis_dir: Path | None = None, num_vis: int = 0) -> dict:
    """Semantic evaluation: one heat map per class (max over its phrases), accumulated in HeatmapMetrics.
    save_dir: also write each image's heat maps in MVP_clean's format (<save_dir>/<stem>/prob_<class>.png +
    run_info.json), so MVP_clean's post-processing / scoring tools can read SAM 3 output directly.
    vis_dir + num_vis: write heat_strip() images for the first num_vis images.
    Always returns per image and class diagnostics (presence, max query prob, max score, max heat, GT pixels)."""
    from sam3.model.sam3_image_processor import Sam3Processor

    split_dir = Path(split_dir)
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_names = [c.get("mvp_class", c["name"]) for c in cats]
    queries = [(phrases or {}).get(c["name"], [c["name"]]) for c in cats]
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    images = coco["images"][:max_images] if max_images else coco["images"]
    proc = Sam3Processor(model)
    acc = HeatmapMetrics(class_names)
    diag = []
    for k, im in enumerate(tqdm(images, desc=f"heatmap eval {split_dir.name}", disable=not progress)):
        img = np.array(Image.open(split_dir / im["file_name"]).convert("RGB"))
        h, w = img.shape[:2]
        valid = no_data_validity_fn(img) if no_data_validity else np.ones((h, w), dtype=bool)
        heats, stats = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            state = proc.set_image(Image.fromarray(img))
            for qs in queries:  # several phrases per class -> max over phrases
                res = [class_heatmap(proc, state, p, (h, w), return_stats=True) for p in qs]
                heats.append(np.max([r[0] for r in res], axis=0))
                stats.append({key: max(r[1][key] for r in res) for key in res[0][1]})
        gts = []
        for i, cat in enumerate(cats):
            gt = _gt_union([a for a in by_img[im["id"]] if a["category_id"] == cat["id"]], h, w)
            acc.update(i, heats[i], gt, valid)
            gts.append(gt & valid)
            stats[i]["max_heat"] = float(heats[i].max())
            diag.append({"file_name": im["file_name"], "class": class_names[i], "gt_pixels": int(gts[i].sum()),
                         **{key: round(v, 4) for key, v in stats[i].items()}})
        if vis_dir is not None and k < num_vis:
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            strip = heat_strip(img, gts, heats, class_names, stats)
            cv2.imwrite(str(Path(vis_dir) / f"{Path(im['file_name']).stem}_heat.jpg"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        if save_dir is not None:
            d = Path(save_dir) / Path(im["file_name"]).stem
            d.mkdir(parents=True, exist_ok=True)
            for name, heat in zip(class_names, heats):
                cv2.imwrite(str(d / f"prob_{name}.png"), np.round(heat * 255).astype(np.uint8))
            (d / "run_info.json").write_text(json.dumps({
                "image": str((split_dir / im["file_name"]).resolve()), "image_size": [w, h], "crop_xywh": None,
                "thresholds": {n: 0.5 for n in class_names}, "source": "SAM 3 heat maps (sam3_eval.py --mode heatmap)"},
                indent=2))
    return {"heatmap": acc, "class_names": class_names, "diagnostics": diag}


# ---------------------------------------------------------------- logging helpers (MVP_clean W&B layout)
def mvp_clean_log_dict(train_m: dict, val_m: dict, train_loss: float, val_loss: float, lr: float) -> dict:
    """Same keys as MVP_clean/training/trainer.py's per-epoch log (lr_backbone/lr_head -> one `lr` here)."""
    return {
        "train/loss": train_loss,
        "train/macro_dice": train_m["macro"]["dice"],
        "train/macro_iou": train_m["macro"]["iou"],
        "val/loss": val_loss,
        "val/macro_iou": val_m["macro"]["iou"],
        "val/macro_dice": val_m["macro"]["dice"],
        "val/freq_weighted_dice": val_m["frequency_weighted"]["dice"],
        "gap/macro_dice_train_minus_val": train_m["macro"]["dice"] - val_m["macro"]["dice"],
        "gap/loss_val_minus_train": val_loss - train_loss,
        "lr": lr,
    }


def wandb_per_class_table(wandb, per_class: dict):
    table = wandb.Table(columns=["class", "iou", "dice", "precision", "recall"])
    for name, m in per_class.items():
        table.add_data(name, m["iou"], m["dice"], m["precision"], m["recall"])
    return table


def wandb_images(wandb, vis: list, class_names: list[str]) -> list:
    labels = {i + 1: n for i, n in enumerate(class_names)}
    return [
        wandb.Image(img, masks={"prediction": {"mask_data": pred, "class_labels": labels},
                                "ground_truth": {"mask_data": gt, "class_labels": labels}})
        for img, gt, pred in vis
    ]


# ---------------------------------------------------------------- CLI
def _panel(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    out = img.copy()
    for v in range(1, int(flat.max()) + 1):
        m = flat == v
        out[m] = (0.5 * out[m] + 0.5 * np.array(COLORS[(v - 1) % len(COLORS)])).astype(np.uint8)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split_dir", required=True, help="split folder with _annotations.coco.json + images")
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="training YAML: its `lora` block (needed with --lora_weights) and `wandb` block")
    p.add_argument("--lora_weights", help="best_lora_weights.pt; leave out for zero-shot SAM 3")
    p.add_argument("--prompts", help="YAML: category name -> list of phrases (default: the category name)")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_vis", type=int, default=10, help="GT|prediction panels to save")
    p.add_argument("--no_validity", action="store_true", help="do not exclude black no-data borders")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_gpu_fraction", type=float, default=None, help="optional hard cap on GPU memory")
    p.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    p.add_argument("--wandb", action="store_true", help="log metrics/table/images as a W&B run (config's wandb block)")
    p.add_argument("--mode", choices=["instances", "heatmap"], default="instances",
                   help="instances (default, unchanged): union of instances with score >= --threshold. "
                        "heatmap: per-class heat map = max over all query slots of score x mask prob (no threshold, "
                        "no NMS); reports metrics at --threshold (or --thresholds), the best-Dice threshold per "
                        "class and threshold-free pixel AP")
    p.add_argument("--thresholds", help="heatmap mode: JSON {class: threshold}, e.g. thresholds_best.json from a "
                                        "val run, to apply on test (choose on val, report on test)")
    p.add_argument("--save_heatmaps", action="store_true",
                   help="heatmap mode: also write <out>/heatmaps/<image>/prob_<class>.png + run_info.json "
                        "(MVP_clean format)")
    p.add_argument("--heat_vis", type=int, default=0,
                   help="heatmap mode: write <out>/vis/<image>_heat.jpg for the first N images (image + GT, "
                        "then one heat map per class with the GT outline, max heat and presence score)")
    p.add_argument("--diagnostics", action="store_true",
                   help="heatmap mode: write <out>/diagnostics.csv (per image and class: GT pixels, presence "
                        "score, max query prob, max instance score, max heat) and print a per-class summary")
    args = p.parse_args()

    sys.path.insert(0, os.getcwd())  # SAM3_LoRA root: lora_layers.py lives there (not pip-installed)
    from sam3.model_builder import build_sam3_image_model

    if args.max_gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction)
    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    model = build_sam3_image_model(bpe_path=args.bpe_path, device="cuda", eval_mode=True, load_from_HF=True)
    if args.lora_weights:
        if "lora" not in cfg:
            raise SystemExit("--lora_weights needs --config with the `lora` block used for training")
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

        lc = cfg["lora"]
        model = apply_lora_to_model(model, LoRAConfig(
            rank=lc["rank"], alpha=lc["alpha"], dropout=0.0, target_modules=lc["target_modules"],
            **{k: lc[k] for k in lc if k.startswith("apply_to_")}))
        load_lora_weights(model, args.lora_weights)
    model.to("cuda").eval()

    phrases = yaml.safe_load(Path(args.prompts).read_text()) if args.prompts else None
    if args.mode == "heatmap":
        _main_heatmap(args, model, phrases, cfg)
        return
    res = evaluate_split(model, args.split_dir, args.max_images, args.threshold, phrases, num_vis=args.num_vis,
                         no_data_validity=not args.no_validity, collect_predictions=True, bf16=args.bf16)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    met = res["metrics"]
    (out / "metrics.json").write_text(json.dumps(met, indent=2))
    (out / "predictions.json").write_text(json.dumps(res["predictions"]))
    with open(out / "metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "iou", "dice", "precision", "recall", "gt_pixels"])
        for name, m in met["per_class"].items():
            wr.writerow([name] + [f"{m[k]:.4f}" for k in ("iou", "dice", "precision", "recall")] + [m["gt_pixels"]])
        for agg in ("macro", "macro_present", "frequency_weighted"):
            wr.writerow([agg] + [f"{met[agg][k]:.4f}" for k in ("iou", "dice", "precision", "recall")] + [""])
    for j, (img, gt, pred) in enumerate(res["vis"]):
        panel = np.hstack([_panel(img, gt), _panel(img, pred)])
        cv2.imwrite(str(out / f"{j:03d}_gt_vs_pred.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    print(f"\n{'class':15s} {'iou':>7s} {'dice/F1':>8s} {'prec':>7s} {'recall':>7s}")
    for name, m in list(met["per_class"].items()) + [(a, met[a]) for a in ("macro", "macro_present", "frequency_weighted")]:
        print(f"{name:15s} {m['iou']:7.4f} {m['dice']:8.4f} {m['precision']:7.4f} {m['recall']:7.4f}")
    print(f"\n{len(res['predictions'])} instances -> {out / 'predictions.json'}; metrics -> {out / 'metrics.csv'}")

    if args.wandb:
        import wandb

        wb = cfg.get("wandb", {})
        split = Path(args.split_dir).name
        name = f"{wb.get('run_name') or 'sam3'}_{'lora' if args.lora_weights else 'zeroshot'}_eval_{split}"
        wandb.init(entity=wb.get("entity"), project=wb.get("project"), name=name, mode=wb.get("mode", "online"),
                   config={"split_dir": args.split_dir, "lora_weights": args.lora_weights, "threshold": args.threshold,
                           "max_images": args.max_images, "prompts": phrases, "train_config": cfg})
        wandb.log({**{f"{split}/{a}_{k}": met[a][k] for a in ("macro", "frequency_weighted")
                      for k in ("iou", "dice", "precision", "recall")},
                   f"{split}/per_class_table": wandb_per_class_table(wandb, met["per_class"]),
                   f"{split}/predictions": wandb_images(wandb, res["vis"], res["class_names"])})
        wandb.finish()


def _write_diagnostics(out: Path, rows: list, class_names: list[str], thr: float) -> None:
    """diagnostics.csv + per-class summary: is the class 'seen' (presence / scores) on images that contain it
    vs images that don't, and how often an instance would pass the instance threshold `thr`."""
    cols = ["file_name", "class", "gt_pixels", "presence", "max_query_prob", "max_score", "max_heat"]
    with open(out / "diagnostics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\ndiagnostics per class (pos = images WITH the class in GT, neg = images WITHOUT it):")
    print(f"{'class':15s} {'n_pos':>5s} {'n_neg':>5s} {'presence pos':>12s} {'presence neg':>12s} "
          f"{'score pos':>9s} {'score neg':>9s} {f'pos>={thr}':>9s} {f'neg>={thr}':>9s}")
    for c in class_names:
        pos = [r for r in rows if r["class"] == c and r["gt_pixels"] > 0]
        neg = [r for r in rows if r["class"] == c and r["gt_pixels"] == 0]
        mean = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
        share = lambda rs: float(np.mean([r["max_score"] >= thr for r in rs])) if rs else float("nan")
        print(f"{c:15s} {len(pos):5d} {len(neg):5d} {mean(pos, 'presence'):12.3f} {mean(neg, 'presence'):12.3f} "
              f"{mean(pos, 'max_score'):9.3f} {mean(neg, 'max_score'):9.3f} {share(pos):9.2f} {share(neg):9.2f}")
    print(f"  presence = SAM 3's 'concept in image' probability; score = best instance score (query x presence);"
          f"\n  pos>={thr}: share of images WITH the class where at least one instance passes {thr} (instance mode"
          f"\n  would predict something); neg>={thr}: same on images WITHOUT the class (false-alarm images)."
          f"\n  -> {out / 'diagnostics.csv'}")


def _main_heatmap(args, model, phrases, cfg: dict) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    res = evaluate_split_heatmap(model, args.split_dir, args.max_images, phrases,
                                 no_data_validity=not args.no_validity, bf16=args.bf16,
                                 save_dir=out / "heatmaps" if args.save_heatmaps else None,
                                 vis_dir=out / "vis" if args.heat_vis else None, num_vis=args.heat_vis)
    if args.diagnostics:
        _write_diagnostics(out, res["diagnostics"], res["class_names"], args.threshold)
    met = res["heatmap"].compute(thresholds, fixed=args.threshold)
    (out / "metrics_heatmap.json").write_text(json.dumps(met, indent=2))
    (out / "thresholds_best.json").write_text(json.dumps(
        {c: m["best_threshold"] for c, m in met["per_class"].items()}, indent=2))
    cols = ["threshold", "iou", "dice", "precision", "recall", "best_threshold", "best_dice", "pixel_ap", "gt_pixels"]
    with open(out / "metrics_heatmap.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class"] + cols)
        for c, m in met["per_class"].items():
            wr.writerow([c] + [m[k] if k == "gt_pixels" else f"{m[k]:.4f}" for k in cols])
        for agg in ("macro", "macro_present"):
            wr.writerow([agg] + [f"{met[agg][k]:.4f}" if k in met[agg] else "" for k in cols])
    src = f"thresholds from {args.thresholds}" if thresholds else f"fixed threshold {args.threshold}"
    print(f"\nheat-map metrics ({src}); best_* = best threshold chosen on THIS split (optimistic)")
    print(f"{'class':15s} {'thr':>5s} {'iou':>7s} {'dice':>7s} {'prec':>7s} {'recall':>7s} "
          f"{'best_thr':>8s} {'best_dice':>9s} {'pixel_AP':>8s} {'gt_pixels':>10s}")
    for c, m in met["per_class"].items():
        print(f"{c:15s} {m['threshold']:5.2f} {m['iou']:7.4f} {m['dice']:7.4f} {m['precision']:7.4f} "
              f"{m['recall']:7.4f} {m['best_threshold']:8.2f} {m['best_dice']:9.4f} {m['pixel_ap']:8.4f} "
              f"{m['gt_pixels']:10d}")
    mp = met["macro_present"]
    print(f"{'macro_present':15s} {'':5s} {mp['iou']:7.4f} {mp['dice']:7.4f} {mp['precision']:7.4f} "
          f"{mp['recall']:7.4f} {'':8s} {mp['best_dice']:9.4f} {mp['pixel_ap']:8.4f}")
    print(f"\n-> {out / 'metrics_heatmap.csv'}, {out / 'thresholds_best.json'}"
          + (f", heat maps in {out / 'heatmaps'}" if args.save_heatmaps else ""))


if __name__ == "__main__":
    main()
