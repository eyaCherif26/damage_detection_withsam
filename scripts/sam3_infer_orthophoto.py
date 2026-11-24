"""SAM 3 (zero-shot, or + fine-tuned LoRA) sliding-window inference on a large orthophoto, with the SAME output
format as MVP_clean's scripts/infer_orthophoto.py, so MVP_clean's post-processing (postprocess_predictions.py:
hysteresis, morphology, merge) and VLM_test/scripts/sam3_postprocess.py can run on the result unchanged.

Per tile (default 1008 px = SAM 3's input size, overlapping) and class: a heat map = per pixel the max over all
SAM 3 query slots of (instance score x mask probability), max over the class's phrases (sam3_eval.class_heatmap:
no instance threshold, no NMS). Overlapping tiles are blended with linear ramps (no seams). Black no-data
regions are set to 0, as in MVP_clean.

Outputs in --output_dir:
  prob_<class>.png       8-bit confidence heat map per class (value / 255 = probability), original pixel grid
  predictions.coco.json  one instance per connected component of (prob >= class threshold), score = mean prob;
                         segmentation RLE (default) or polygon rings (--segmentation_format polygon)
  overlay.png            masks over the image (+ overlay_preview.jpg downscaled if the image is large)
  run_info.json          all settings, class list + thresholds, class colours, predicted pixel share

Run from the SAM3_LoRA repo root in sam3_lora_env:
  python ../scripts/sam3_infer_orthophoto.py --image <orthophoto> --output_dir <dir> \
      --config ../configs/cif_24gb.yaml --lora_weights ../outputs/sam3_cif_24gb/best_lora_weights.pt \
      --thresholds ../outputs/heat_lora_valid/thresholds_best.json
  (zero-shot SAM 3: leave out --config / --lora_weights)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/: sam3_eval
from sam3_eval import COLORS, class_heatmap, no_data_validity  # noqa: E402

# MVP_clean class -> text prompt(s); must match the category names the LoRA was trained with
# (VLM_test/scripts/cif_to_sam3_coco.py DEFAULT_PROMPTS). Override with --prompts.
DEFAULT_PROMPTS = {
    "Cracks": ["crack"],
    "Wetspot": ["wet spot"],
    "Spalling": ["spalling"],
    "Rust": ["rust"],
    "Efflorescence": ["efflorescence"],
}


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def blend_weight(th: int, tw: int, overlap: int) -> np.ndarray:
    """Tile weight: 1 inside, linear ramp over `overlap` px at each border, so overlapping tiles fade into each
    other. The accumulated map is divided by the summed weights, so single-tile areas keep their exact values."""
    def ramp(n: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float32)
        k = min(overlap, n // 2)
        if k > 0:
            v = (np.arange(k, dtype=np.float32) + 1) / (k + 1)
            r[:k] = v
            r[-k:] = v[::-1]
        return r
    return np.outer(ramp(th), ramp(tw))


def export_predictions(binary: dict, probs: dict, classes: list[str], file_name: str, h: int, w: int,
                       min_area: int, fmt: str) -> dict:
    """MVP_clean-style predictions COCO: one annotation per connected component (8-connectivity)."""
    anns = []
    buf = np.zeros((h, w), dtype=np.uint8) if fmt == "rle" else None
    for ci, c in enumerate(classes):
        lab, n = ndimage.label(binary[c], structure=np.ones((3, 3), dtype=bool))
        areas = np.bincount(lab.ravel(), minlength=n + 1)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None or areas[i] < min_area:
                continue
            sy, sx = sl
            comp = lab[sy, sx] == i
            if fmt == "rle":
                buf[sy, sx] = comp
                seg = mask_utils.encode(np.asfortranarray(buf))
                buf[sy, sx] = 0
                seg["counts"] = seg["counts"].decode("ascii")
            else:  # polygon rings (outer contours and holes, one flat ring each), original pixel coordinates
                cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                seg = [(cnt.reshape(-1, 2) + [sx.start, sy.start]).astype(float).ravel().tolist()
                       for cnt in cs if len(cnt) >= 3]
                if not seg:
                    continue
            anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": ci + 1, "segmentation": seg,
                         "area": int(areas[i]), "bbox": [sx.start, sy.start, sx.stop - sx.start, sy.stop - sy.start],
                         "score": round(float(probs[c][sy, sx][comp].mean()), 4), "iscrowd": 0})
    return {"images": [{"id": 1, "file_name": file_name, "width": w, "height": h}],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)], "annotations": anns}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--config", help="training YAML (its `lora` block); needed with --lora_weights")
    p.add_argument("--lora_weights", help="best_lora_weights.pt; leave out for zero-shot SAM 3")
    p.add_argument("--prompts", help="YAML: class -> phrase or list of phrases (default: the 5 CIF training prompts)")
    p.add_argument("--thresholds", help="JSON {class: threshold}, e.g. thresholds_best.json from the heat-map "
                                        "validation run (default 0.5 for every class)")
    p.add_argument("--tile_size", type=int, default=1008, help="SAM 3's input size")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--scale", type=float, default=1.0,
                   help="resample the image by this factor before tiling (GSD matching, as in MVP_clean); "
                        "outputs stay in original pixel coordinates")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="process only this window (outputs in crop coordinates, as in MVP_clean)")
    p.add_argument("--min_area", type=int, default=16, help="drop predicted components smaller than this (px)")
    p.add_argument("--segmentation_format", choices=["rle", "polygon"], default="rle")
    p.add_argument("--preview_max_side", type=int, default=4096, help="overlay_preview.jpg size; 0 disables")
    p.add_argument("--max_memory_gb", type=float, default=8.0, help="refuse larger prob-map accumulators ...")
    p.add_argument("--allow_large_memory", action="store_true", help="... unless this is given")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_gpu_fraction", type=float, default=None)
    p.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = p.parse_args()
    if args.tile_size <= args.overlap:
        raise SystemExit("--overlap must be smaller than --tile_size")

    # ---- classes, prompts, thresholds
    prompts = DEFAULT_PROMPTS
    if args.prompts:
        raw = yaml.safe_load(Path(args.prompts).read_text())
        prompts = {c: ([v] if isinstance(v, str) else list(v)) for c, v in raw.items()}
    classes = list(prompts)
    thr_file = json.loads(Path(args.thresholds).read_text()) if args.thresholds else {}
    thresholds = {c: float(thr_file.get(c, 0.5)) for c in classes}
    missing = [c for c in classes if args.thresholds and c not in thr_file]
    if missing:
        print(f"[sam3_infer] WARNING: no threshold for {missing} in {args.thresholds}; using 0.5")

    # ---- image (crop / scale as in MVP_clean)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    full_h, full_w = image.shape[:2]
    crop_origin = (0, 0)
    if args.crop:
        x, y, cw, ch = args.crop
        if cw <= 0 or ch <= 0 or x < 0 or y < 0 or x + cw > full_w or y + ch > full_h:
            raise SystemExit(f"--crop {args.crop} is outside the image ({full_w}x{full_h}) or empty")
        image = np.ascontiguousarray(image[y:y + ch, x:x + cw])
        crop_origin = (x, y)
    h, w = image.shape[:2]
    model_input = image
    if args.scale != 1.0:
        model_input = cv2.resize(image, (round(w * args.scale), round(h * args.scale)), interpolation=cv2.INTER_AREA)
    H, W = model_input.shape[:2]
    need_gb = (len(classes) + 1) * H * W * 4 / 1e9
    if need_gb > args.max_memory_gb and not args.allow_large_memory:
        raise SystemExit(f"prob-map accumulators need ~{need_gb:.1f} GB RAM (> --max_memory_gb {args.max_memory_gb}); "
                         "use --crop / --scale, or --allow_large_memory")
    step = args.tile_size - args.overlap
    tiles = [(x, y) for y in tile_starts(H, args.tile_size, step) for x in tile_starts(W, args.tile_size, step)]
    print(f"[sam3_infer] {args.image} ({full_w}x{full_h})" + (f", crop {args.crop}" if args.crop else "")
          + (f", scale {args.scale} -> {W}x{H}" if args.scale != 1.0 else "")
          + f", {len(tiles)} tiles of {args.tile_size} (overlap {args.overlap}), classes {classes}")

    # ---- model (SAM 3 + optional LoRA, as in sam3_eval.py)
    sys.path.insert(0, os.getcwd())  # SAM3_LoRA root
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.max_gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction)
    model = build_sam3_image_model(bpe_path=args.bpe_path, device="cuda", eval_mode=True, load_from_HF=True)
    if args.lora_weights:
        if not args.config:
            raise SystemExit("--lora_weights needs --config (the training YAML with its `lora` block)")
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

        lc = yaml.safe_load(Path(args.config).read_text())["lora"]
        model = apply_lora_to_model(model, LoRAConfig(
            rank=lc["rank"], alpha=lc["alpha"], dropout=0.0, target_modules=lc["target_modules"],
            **{k: lc[k] for k in lc if k.startswith("apply_to_")}))
        load_lora_weights(model, args.lora_weights)
    model.to("cuda").eval()
    proc = Sam3Processor(model)

    # ---- sliding window
    acc = np.zeros((len(classes), H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    t0 = time.time()
    for k, (x, y) in enumerate(tiles, start=1):
        tile = model_input[y:y + args.tile_size, x:x + args.tile_size]
        th, tw = tile.shape[:2]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            state = proc.set_image(Image.fromarray(np.ascontiguousarray(tile)))
            heats = [np.max([class_heatmap(proc, state, ph, (th, tw)) for ph in prompts[c]], axis=0) for c in classes]
        wgt = blend_weight(th, tw, args.overlap)
        acc[:, y:y + th, x:x + tw] += np.stack(heats) * wgt
        wsum[y:y + th, x:x + tw] += wgt
        if k == 1 or k % 10 == 0 or k == len(tiles):
            el = time.time() - t0
            print(f"[sam3_infer] tile {k}/{len(tiles)} ({el / k:.1f} s/tile, ~{el / k * (len(tiles) - k) / 60:.1f} min left)",
                  flush=True)
    prob = acc / np.maximum(wsum, 1e-6)
    del acc, wsum
    if args.scale != 1.0:
        prob = np.stack([cv2.resize(prob[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(len(classes))])
    prob *= no_data_validity(image)[None]  # black no-data borders -> 0, as in MVP_clean
    probs = {c: prob[i] for i, c in enumerate(classes)}
    binary = {c: probs[c] >= thresholds[c] for c in classes}

    # ---- outputs (MVP_clean layout)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    colors = {c: list(COLORS[i % len(COLORS)]) for i, c in enumerate(classes)}
    overlay = image.copy()
    for c in classes:
        cv2.imwrite(str(out / f"prob_{c}.png"), np.round(probs[c] * 255).astype(np.uint8))
        m = binary[c]
        overlay[m] = (0.5 * overlay[m] + 0.5 * np.array(colors[c])).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "overlay.png"), overlay_bgr)
    if args.preview_max_side and max(h, w) > args.preview_max_side:
        f = args.preview_max_side / max(h, w)
        cv2.imwrite(str(out / "overlay_preview.jpg"),
                    cv2.resize(overlay_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA))
    coco = export_predictions(binary, probs, classes, args.image.name, h, w, args.min_area, args.segmentation_format)
    (out / "predictions.coco.json").write_text(json.dumps(coco))
    share = {c: round(float(binary[c].mean()), 4) for c in classes}
    (out / "run_info.json").write_text(json.dumps({
        "image": str(args.image), "image_size": [full_w, full_h], "crop_xywh": args.crop,
        "crop_origin": list(crop_origin), "scale": args.scale, "tile_size": args.tile_size, "overlap": args.overlap,
        "model": "SAM 3" + (" + LoRA" if args.lora_weights else " (zero-shot)"), "config": args.config,
        "checkpoint": args.lora_weights, "prompts": prompts, "thresholds": thresholds, "class_colors_rgb": colors,
        "predicted_pixel_share": share, "min_area": args.min_area,
        "segmentation_format": args.segmentation_format, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[sam3_infer] wrote outputs to {out}; {len(coco['annotations'])} instances; predicted pixel share {share}")
    print(f"[sam3_infer] overlay colours (RGB): {colors}")


if __name__ == "__main__":
    main()
