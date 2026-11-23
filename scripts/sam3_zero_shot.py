"""Zero-shot SAM 3 (no LoRA) on a SAM3_LoRA-layout split, one or several text phrases per category.

Per image and phrase SAM 3 returns instances (mask, box, score). Instances of all phrases of a category are
merged into that category. Outputs in --out:
  predictions.json        COCO results list: image_id, file_name, category_id, category, prompt, score,
                          bbox [x, y, w, h], area, segmentation (RLE, original image size)
  <image>_gt_vs_pred.jpg  ground truth | prediction (masks + boxes + scores)
and prints per-phrase instance counts, per-class pixel IoU (union of instances vs union of GT) and peak GPU memory.

Run from the SAM3_LoRA repo root (needs its `sam3` package and BPE file):
    cd SAM3_LoRA && python ../scripts/sam3_zero_shot.py --split_dir ../data/cif_sam3_smoke/valid \
        --out ../outputs/sam3_zero_shot --max_images 1 --bf16 [--prompts ../configs/sam3_prompts_cif.yaml]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 255, 0)]  # RGB


def gt_union(anns: list[dict], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for a in anns:
        seg = a["segmentation"]
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
        m |= mask_utils.decode(rle).astype(bool)
    return m


def draw(img: np.ndarray, masks: dict[int, np.ndarray], boxes: list[tuple] = ()) -> np.ndarray:
    out = img.copy()
    for cid, m in masks.items():
        out[m] = (0.5 * out[m] + 0.5 * np.array(COLORS[(cid - 1) % len(COLORS)])).astype(np.uint8)
    for cid, (x0, y0, x1, y1), label in boxes:
        c = COLORS[(cid - 1) % len(COLORS)]
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
        cv2.putText(out, label, (int(x0), max(int(y0) - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split_dir", help="labelled split: folder with _annotations.coco.json + images")
    src.add_argument("--image_dir", help="unlabelled folder of images (needs --prompts; no GT scoring)")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", default=None, help="YAML: category name -> list of phrases (default: category name)")
    p.add_argument("--max_images", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    p.add_argument("--bf16", action="store_true", help="run under bf16 autocast (roughly halves activation memory)")
    p.add_argument("--max_gpu_fraction", type=float, default=0.8,
                   help="hard cap on the GPU memory PyTorch may use -> clean OOM error instead of a driver crash")
    args = p.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.split_dir:
        split = Path(args.split_dir)
        coco = json.loads((split / "_annotations.coco.json").read_text())
    else:
        if not args.prompts:
            raise SystemExit("--image_dir needs --prompts (the phrase file defines the categories)")
        split = Path(args.image_dir)
        files = sorted(f.name for f in split.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        names = list(yaml.safe_load(Path(args.prompts).read_text()))
        coco = {"images": [{"id": i, "file_name": f} for i, f in enumerate(files)], "annotations": [],
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    labelled = bool(args.split_dir)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    phrases = {cid: [name] for cid, name in cats.items()}
    if args.prompts:
        table = yaml.safe_load(Path(args.prompts).read_text())
        unknown = set(table) - set(cats.values())
        if unknown:
            raise SystemExit(f"--prompts has categories not in the dataset: {sorted(unknown)}; dataset: {sorted(cats.values())}")
        phrases = {cid: table.get(name, [name]) for cid, name in cats.items()}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    model = build_sam3_image_model(bpe_path=args.bpe_path, device="cuda", eval_mode=True, load_from_HF=True)
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    torch.cuda.reset_peak_memory_stats()

    inter, union, per_phrase, results = defaultdict(int), defaultdict(int), defaultdict(int), []
    for im in coco["images"][: args.max_images]:
        img = np.array(Image.open(split / im["file_name"]).convert("RGB"))
        h, w = img.shape[:2]
        gt, pred, boxes, log = {}, {}, [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            state = proc.set_image(Image.fromarray(img))
            for cid, name in cats.items():
                cls_mask = np.zeros((h, w), bool)
                for phrase in phrases[cid]:
                    state = proc.set_text_prompt(prompt=phrase, state=state)
                    masks = state["masks"].squeeze(1).cpu().numpy()
                    for m, b, s in zip(masks, state["boxes"].float().cpu().tolist(), state["scores"].float().cpu().tolist()):
                        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode("ascii")
                        x0, y0, x1, y1 = b
                        results.append({"image_id": im["id"], "file_name": im["file_name"], "category_id": cid,
                                        "category": name, "prompt": phrase, "score": round(s, 4),
                                        "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                                        "area": int(m.sum()), "segmentation": rle})
                        boxes.append((cid, b, f"{phrase} {s:.2f}"))
                        cls_mask |= m
                    per_phrase[(name, phrase)] += len(masks)
                    log.append(f"{phrase}: {len(masks)}")
                    proc.reset_all_prompts(state)
                g = gt_union([a for a in by_img[im["id"]] if a["category_id"] == cid], h, w)
                inter[name] += int((cls_mask & g).sum())
                union[name] += int((cls_mask | g).sum())
                if g.any():
                    gt[cid] = g
                if cls_mask.any():
                    pred[cid] = cls_mask
        panel = np.hstack([draw(img, gt), draw(img, pred, boxes)]) if labelled else draw(img, pred, boxes)
        cv2.imwrite(str(out / f"{Path(im['file_name']).stem}_gt_vs_pred.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"{im['file_name']}: GT {sorted(cats[c] for c in gt)} | instances per phrase: " + ", ".join(log)
              + f" | peak GPU so far {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    (out / "predictions.json").write_text(json.dumps(results))
    print(f"\n{len(results)} instances -> {out / 'predictions.json'}")
    print("instances per phrase (all images):")
    for (name, phrase), n in per_phrase.items():
        print(f"  {name:15s} {phrase!r:28s} {n}")
    if labelled:
        print("per-class pixel IoU (merged phrases, over these images):")
        for name in cats.values():
            print(f"  {name:15s} {inter[name] / union[name]:.3f}" if union[name] else f"  {name:15s} n/a (absent, no FP)")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("colors (RGB):", {cats[c]: COLORS[(c - 1) % len(COLORS)] for c in cats})


if __name__ == "__main__":
    main()
