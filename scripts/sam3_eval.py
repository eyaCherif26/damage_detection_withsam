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


if __name__ == "__main__":
    main()
