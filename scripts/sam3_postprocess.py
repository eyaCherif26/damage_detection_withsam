"""SAM 3 (zero-shot) post-processing of MVP_clean predictions, three independent stages:
  1. remove damage lying on non-structural objects (ladder, fender, chain, ...)   -> variant s1
  2. refine the remaining damage masks with SAM 3's box/point predictor           -> variant s12
  3. add missed damage similar to the confirmed damage (exemplar search)          -> variant s123
File in -> file out, like MVP_clean's scripts/postprocess_predictions.py; MVP_clean's outputs are never modified.

Input   --pred_root/<image>/   one folder per image from MVP_clean scripts/infer_orthophoto.py
                               (prob_<class>.png, run_info.json; no MVP_clean post-processing)
        --images_dir           the original images (looked up by the file name recorded in run_info.json)
Output  --pred_root/<image>/sam3_pp/
          base/ s1/ s12/ s123/   mask_<class>.png (0/255, original size); base = MVP_clean prob >= its thresholds
          objects.png            union of the SAM 3 object masks used by stage 1
        --pred_root/<image>/sam3_filtered/   (stage 1) prob_<class>.png zeroed on objects + removed components,
          and run_info.json -> input for MVP_clean's hysteresis: postprocess_predictions.py --input_dir <this>
          predictions.coco.json  last variant: one instance per connected component (RLE; score = mean MVP_clean prob)
          overlay.jpg            last variant over the image, MVP_clean class colours, objects tinted grey
          sam3_info.json         settings + every decision (removed / refined / added components)

Run from the SAM3_LoRA repo root in sam3_lora_env (24 GB GPU):
    python ../scripts/sam3_postprocess.py --pred_root <mvp outputs> --images_dir <images> --config ../configs/sam3_pp.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

CONN8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------- helpers
def load_mvp_outputs(d: Path) -> tuple[dict, list[str], dict, dict]:
    info = json.loads((d / "run_info.json").read_text())
    thresholds = info["thresholds"]
    classes = list(thresholds)
    probs = {}
    for c in classes:
        p = cv2.imread(str(d / f"prob_{c}.png"), cv2.IMREAD_GRAYSCALE)
        if p is None:
            raise FileNotFoundError(d / f"prob_{c}.png")
        probs[c] = p.astype(np.float32) / 255.0
    return info, classes, thresholds, probs


def load_image(info: dict, images_dir: Path) -> np.ndarray:
    path = images_dir / Path(info["image"]).name
    if not path.exists():
        path = Path(info["image"])
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"image not found in --images_dir or at run_info's path: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if info.get("crop_xywh"):  # MVP_clean was run on a window only
        x, y, cw, ch = info["crop_xywh"]
        img = np.ascontiguousarray(img[y:y + ch, x:x + cw])
    return img


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def windows(h: int, w: int, tile: int, overlap: int, trigger: float) -> list[tuple[int, int, int, int]]:
    """(x, y, width, height) windows for SAM 3: the whole image if small enough, else overlapping tiles."""
    if max(h, w) <= tile * trigger:
        return [(0, 0, w, h)]
    step = tile - overlap
    return [(x, y, min(tile, w), min(tile, h)) for y in _starts(h, tile, step) for x in _starts(w, tile, step)]


def query(proc, state: dict, phrase: str | None, exemplar_boxes=()) -> tuple[np.ndarray, np.ndarray]:
    """One SAM 3 concept query on the image set in `state`: text phrase and/or positive exemplar boxes
    (cx, cy, w, h normalised to the window). Returns (N, h, w) bool masks and (N,) scores."""
    if phrase is not None:
        state = proc.set_text_prompt(prompt=phrase, state=state)
    for box in exemplar_boxes:
        state = proc.add_geometric_prompt(box, True, state)
    masks = state["masks"].squeeze(1).cpu().numpy().astype(bool)
    scores = state["scores"].float().cpu().numpy()
    proc.reset_all_prompts(state)
    return masks, scores


def components(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    lab, n = ndimage.label(mask, structure=CONN8)
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, n, areas


# ---------------------------------------------------------------- stage 1
def detect_objects(proc, img: np.ndarray, cfg: dict, wins: list) -> tuple[np.ndarray, dict, dict, dict]:
    """Object prompts -> one union mask (damage there is removed). Look-alike prompts (optional) -> kept as
    separate instances (y0, x0, cropped mask) per phrase, used by remove_look_alikes()."""
    h, w = img.shape[:2]
    obj = np.zeros((h, w), dtype=bool)
    look = cfg.get("look_alike_prompts") or {}
    phrases = list(dict.fromkeys(list(cfg["prompts"]) + list(look)))
    found = {p: 0 for p in phrases}
    best = {p: 0.0 for p in phrases}  # highest score per prompt (>= log floor; 0.0 = nothing above it)
    instances = {p: [] for p in look}
    thr = cfg["score_threshold"]
    phrase_thr = {p: (look_alike_spec(look[p], cfg)[3] if p in look else thr) for p in phrases}
    # no log_min_score / per-phrase thresholds = processor threshold as before
    proc.set_confidence_threshold(min([thr, cfg.get("log_min_score", thr)] + list(phrase_thr.values())))
    for x, y, tw, th in wins:
        state = proc.set_image(Image.fromarray(img[y:y + th, x:x + tw]))
        for prompt in phrases:
            masks, scores = query(proc, state, prompt)
            if len(scores):
                best[prompt] = max(best[prompt], float(scores.max()))
            keep = scores >= phrase_thr[prompt]
            if not keep.any():
                continue
            found[prompt] += int(keep.sum())
            if prompt in look:
                for m in masks[keep]:
                    ys, xs = np.nonzero(m)
                    if len(ys):
                        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                        instances[prompt].append((y + y0, x + x0, m[y0:y1, x0:x1].copy()))
            else:
                obj[y:y + th, x:x + tw] |= masks[keep].any(0)
    d = int(cfg.get("dilate_px", 0))
    if d > 0:
        obj = cv2.dilate(obj.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0
    return obj, found, {p: round(s, 3) for p, s in best.items()}, instances


def look_alike_spec(value, c1: dict) -> tuple[list, float, float | None, float]:
    """look_alike_prompts entry -> (classes, min_overlap, max_area_ratio or None, score_threshold).
    List form `phrase: [Cracks]` = global settings (original behaviour); dict form
    `phrase: {classes: [...], min_overlap: .., max_area_ratio: null, score_threshold: ..}` overrides per phrase
    (max_area_ratio: null = no size condition)."""
    g_overlap = c1.get("look_alike_min_overlap", 0.5)
    g_ratio = c1.get("look_alike_max_area_ratio", 4.0)
    if isinstance(value, dict):
        return (list(value["classes"]), value.get("min_overlap", g_overlap),
                value.get("max_area_ratio", g_ratio), value.get("score_threshold", c1["score_threshold"]))
    return list(value), g_overlap, g_ratio, c1["score_threshold"]


def remove_look_alikes(mask: np.ndarray, instances: list, min_overlap: float, max_area_ratio: float | None,
                       details: list | None = None) -> tuple[np.ndarray, int]:
    """Remove damage components that essentially ARE a look-alike instance (e.g. a branch shadow predicted as a
    crack): >= min_overlap of the component inside one instance whose area <= max_area_ratio x component area
    (max_area_ratio None = no size condition). Damage inside a much larger instance (a big shaded area) is kept
    unless the size condition is off. `details` (optional) collects per-component numbers for the log."""
    lab, n, areas = components(mask)
    if n == 0 or not instances:
        return mask, 0
    drop, best = set(), {}
    for y0, x0, m in instances:
        sub = lab[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m]
        if not sub.size:
            continue
        inter = np.bincount(sub, minlength=n + 1)
        inter[0] = 0
        inst_area = int(m.sum())
        for i in np.flatnonzero(inter):
            frac, ratio = inter[i] / areas[i], inst_area / areas[i]
            ok = frac >= min_overlap and (max_area_ratio is None or ratio <= max_area_ratio)
            if ok:
                drop.add(int(i))
            if int(i) not in best or frac > best[int(i)][0]:
                best[int(i)] = (frac, ratio, ok)
    if details is not None:
        for i, (frac, ratio, _) in best.items():
            details.append({"component_area": int(areas[i]), "best_overlap": round(float(frac), 3),
                            "instance_to_component_area": round(float(ratio), 1), "removed": i in drop})
    if not drop:
        return mask, 0
    return mask & ~np.isin(lab, sorted(drop)), len(drop)


def remove_on_objects(mask: np.ndarray, obj: np.ndarray, frac: float) -> tuple[np.ndarray, int, int]:
    """Drop components with >= frac of their pixels on objects; remove remaining on-object pixels too."""
    lab, n, areas = components(mask)
    if n == 0:
        return mask, 0, 0
    on = np.bincount(lab.ravel(), weights=obj.ravel().astype(np.float64), minlength=n + 1)
    drop = np.flatnonzero(on[1:] >= frac * areas[1:]) + 1
    out = mask & ~np.isin(lab, drop) & ~obj
    return out, int(len(drop)), int(mask.sum() - out.sum())


# ---------------------------------------------------------------- stage 2
def pick_points(prob: np.ndarray, comp: np.ndarray, k: int, min_dist: int) -> np.ndarray:
    """Up to k most confident pixels of the component, at least min_dist apart -> (n, 2) x, y."""
    p = np.where(comp, prob, -1.0)
    pts = []
    for _ in range(k):
        idx = int(np.argmax(p))
        if p.flat[idx] < 0:
            break
        yy, xx = divmod(idx, p.shape[1])
        pts.append([xx, yy])
        p[max(0, yy - min_dist):yy + min_dist + 1, max(0, xx - min_dist):xx + min_dist + 1] = -1.0
    return np.array(pts, dtype=np.float64)


@torch.inference_mode()
def refine(proc, model, img: np.ndarray, masks: dict, probs: dict, thresholds: dict, obj: np.ndarray, cfg: dict) -> tuple[dict, list]:
    """Box/point refinement via SAM 3's interactive predictor. It shares the detector's image encoder, so the
    crop is encoded with proc.set_image and predicted with model.predict_inst (sam3/model/sam3_image.py)."""
    h, w = img.shape[:2]
    out, log = {}, []
    for c, m in masks.items():
        if c not in cfg["classes"]:
            out[c] = m
            continue
        new = m.copy()
        lab, n, areas = components(m)
        ids = [i for i in np.argsort(-areas[1:]) + 1 if areas[i] >= cfg["min_area_px"]][: cfg["max_components_per_class"]]
        slices = ndimage.find_objects(lab)
        for i in ids:
            sy, sx = slices[i - 1]
            y0, y1, x0, x1 = sy.start, sy.stop, sx.start, sx.stop
            margin = max(cfg["crop_margin_min_px"], int(cfg["crop_margin_frac"] * max(y1 - y0, x1 - x0)))
            cy0, cy1, cx0, cx1 = max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin)
            comp = lab[cy0:cy1, cx0:cx1] == i
            prob = probs[c][cy0:cy1, cx0:cx1]
            pts = pick_points(prob, comp, cfg["num_pos_points"], max(3, int(0.15 * max(y1 - y0, x1 - x0))))
            state = proc.set_image(Image.fromarray(np.ascontiguousarray(img[cy0:cy1, cx0:cx1])))
            cand, quality, _ = model.predict_inst(
                state,
                point_coords=pts if len(pts) else None,
                point_labels=np.ones(len(pts)) if len(pts) else None,
                box=np.array([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], dtype=np.float64),
                multimask_output=cfg["multimask_output"],
            )
            cand = cand > 0.5
            ious = [float((cm & comp).sum() / max(1, (cm | comp).sum())) for cm in cand]
            b = int(np.argmax(ious))
            mean_p = float(prob[cand[b]].mean()) if cand[b].any() else 0.0
            ok = ious[b] >= cfg["min_iou_with_original"] and mean_p >= cfg["min_mean_prob_frac"] * thresholds[c]
            if ok:
                region = new[cy0:cy1, cx0:cx1]  # view: edits `new` in place
                region[comp] = False
                region[cand[b]] = True
            log.append({"class": c, "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)], "area": int(areas[i]),
                        "iou_with_original": round(ious[b], 3), "sam_quality": round(float(quality[b]), 3),
                        "mean_prob": round(mean_p, 3), "new_area": int(cand[b].sum()), "accepted": bool(ok)})
        out[c] = new & ~obj
    return out, log


# ---------------------------------------------------------------- stage 3
def add_similar(proc, img: np.ndarray, masks: dict, probs: dict, obj: np.ndarray, cfg: dict, phrases: dict,
                obj_frac: float, wins: list) -> tuple[dict, list]:
    out = {c: m.copy() for c, m in masks.items()}
    log = []
    proc.set_confidence_threshold(cfg["score_threshold"])
    for x, y, tw, th in wins:
        sl = (slice(y, y + th), slice(x, x + tw))
        state = None
        for c in masks:
            if c not in phrases:
                continue
            known = masks[c][sl]
            lab, n, areas = components(known)
            ids = [i for i in np.argsort(-areas[1:]) + 1 if areas[i] >= cfg["min_area_px"]][: cfg["max_exemplars"]]
            if not ids:
                continue
            if state is None:
                state = proc.set_image(Image.fromarray(img[sl]))
            slices = ndimage.find_objects(lab)
            boxes = []
            for i in ids:
                sy, sx = slices[i - 1]
                boxes.append([(sx.start + sx.stop) / 2 / tw, (sy.start + sy.stop) / 2 / th,
                              (sx.stop - sx.start) / tw, (sy.stop - sy.start) / th])
            inst, scores = query(proc, state, phrases[c] if cfg["use_text"] else None, boxes)
            o, p = obj[sl], probs[c][sl]
            for im, s in zip(inst, scores):
                area = int(im.sum())
                if area < cfg["min_area_px"]:
                    continue
                if (im & known).sum() / area >= cfg["max_overlap_with_existing"]:
                    continue  # an already predicted (e.g. exemplar) region
                if (im & o).sum() / area >= obj_frac:
                    continue  # lies on an object
                mean_p = float(p[im].mean())
                if mean_p < cfg["min_mean_prob"]:
                    continue
                add = im & ~o & ~out[c][sl]
                out[c][sl] |= add
                ys, xs = np.nonzero(im)
                log.append({"class": c, "bbox_xyxy": [int(x + xs.min()), int(y + ys.min()), int(x + xs.max() + 1),
                            int(y + ys.max() + 1)], "area": area, "added_px": int(add.sum()),
                            "score": round(float(s), 3), "mean_prob": round(mean_p, 3)})
    return out, log


# ---------------------------------------------------------------- outputs
def save_variant(dst: Path, masks: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for c, m in masks.items():
        cv2.imwrite(str(dst / f"mask_{c}.png"), m.astype(np.uint8) * 255)


def export_coco(masks: dict, probs: dict, file_name: str, h: int, w: int, min_area: int) -> dict:
    classes = list(masks)
    anns, buf = [], np.zeros((h, w), dtype=np.uint8)
    for ci, c in enumerate(classes):
        lab, n, areas = components(masks[c])
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None or areas[i] < min_area:
                continue
            sy, sx = sl
            comp = lab[sy, sx] == i
            buf[sy, sx] = comp
            rle = mask_utils.encode(np.asfortranarray(buf))
            buf[sy, sx] = 0
            rle["counts"] = rle["counts"].decode("ascii")
            anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": ci + 1, "segmentation": rle,
                         "area": int(areas[i]), "bbox": [sx.start, sy.start, sx.stop - sx.start, sy.stop - sy.start],
                         "score": round(float(probs[c][sy, sx][comp].mean()), 4), "iscrowd": 0})
    return {"images": [{"id": 1, "file_name": file_name, "width": w, "height": h}],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)], "annotations": anns}


def overlay(img: np.ndarray, masks: dict, obj: np.ndarray | None, colors: dict) -> np.ndarray:
    out = img.copy()
    if obj is not None:
        out[obj] = (0.6 * out[obj] + 0.4 * 128).astype(np.uint8)
    for c, m in masks.items():
        col = np.array(colors.get(c, [255, 0, 0]))
        out[m] = (0.5 * out[m] + 0.5 * col).astype(np.uint8)
    return out


# ---------------------------------------------------------------- main
def process_image(d: Path, args, cfg: dict, proc, predictor) -> dict:
    t0 = time.time()
    info, classes, thresholds, probs = load_mvp_outputs(d)
    img = load_image(info, Path(args.images_dir))
    h, w = img.shape[:2]
    if probs[classes[0]].shape != (h, w):
        raise ValueError(f"image {w}x{h} and prob maps {probs[classes[0]].shape[::-1]} differ in size")
    s, c1, c2, c3 = cfg["sam3"], cfg["stage1_objects"], cfg["stage2_refine"], cfg["stage3_add"]
    wins = windows(h, w, s["tile_size"], s["tile_overlap"], s["tile_trigger"])
    dst = d / "sam3_pp"
    log: dict = {"image": info["image"], "size_wh": [w, h], "windows": len(wins), "classes": classes,
                 "thresholds": thresholds, "config": cfg}

    masks = {c: probs[c] >= thresholds[c] for c in classes}
    save_variant(dst / "base", masks)
    obj, last = None, "base"
    obj_frac = c1["component_overlap_frac"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(s.get("bf16", False))):
        if c1["enabled"]:
            obj, found, best_scores, look_insts = detect_objects(proc, img, c1, wins)
            cv2.imwrite(str(dst / "objects.png"), obj.astype(np.uint8) * 255)
            stage1 = {}
            for c in classes:
                masks[c], n_comp, n_px = remove_on_objects(masks[c], obj, obj_frac)
                stage1[c] = {"components_removed": n_comp, "pixels_removed": n_px}
            look_cfg = c1.get("look_alike_prompts") or {}
            if look_cfg:  # debug image: every look-alike instance SAM 3 found (e.g. shadows)
                la = np.zeros((h, w), dtype=bool)
                for insts in look_insts.values():
                    for y0, x0, m in insts:
                        la[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
                cv2.imwrite(str(dst / "look_alikes.png"), la.astype(np.uint8) * 255)
            la_details = {}
            for phrase, spec in look_cfg.items():
                look_classes, min_ov, max_ratio, _ = look_alike_spec(spec, c1)
                for c in look_classes:
                    if c not in masks:
                        print(f"[sam3_pp] WARNING: look_alike_prompts '{phrase}': class {c!r} not in {classes}")
                        continue
                    det = la_details.setdefault(f"{phrase} -> {c}", [])
                    masks[c], n_la = remove_look_alikes(masks[c], look_insts[phrase], min_ov, max_ratio, det)
                    stage1[c]["look_alike_components_removed"] = stage1[c].get("look_alike_components_removed", 0) + n_la
            log["stage1"] = {"objects_found_per_prompt": found, "best_score_per_prompt": best_scores,
                             "object_pixel_share": round(float(obj.mean()), 4),
                             "removed": stage1}
            if la_details:  # per damage component touched by a look-alike: best overlap, size ratio, removed?
                log["stage1"]["look_alike_details"] = {k: v[:300] for k, v in la_details.items()}
            save_variant(dst / "s1", masks)
            last = "s1"
            # Filtered copy of MVP_clean's outputs for MVP_clean's own hysteresis post-processing
            # (scripts/postprocess_predictions.py --input_dir <image>/sam3_filtered): probability set to 0
            # on objects and on removed damage components, so hysteresis cannot grow back onto them.
            filt = d / "sam3_filtered"
            filt.mkdir(parents=True, exist_ok=True)
            for c in classes:
                removed = (probs[c] >= thresholds[c]) & ~masks[c]
                pc = probs[c].copy()
                pc[obj | removed] = 0.0
                cv2.imwrite(str(filt / f"prob_{c}.png"), np.round(pc * 255).astype(np.uint8))
            (filt / "run_info.json").write_text(json.dumps({**info, "sam3_filtered_from": str(d)}, indent=2))
        no_obj = obj if obj is not None else np.zeros((h, w), dtype=bool)
        if c2["enabled"]:
            masks, refine_log = refine(proc, predictor, img, masks, probs, thresholds, no_obj, c2)
            log["stage2"] = {"components_tried": len(refine_log),
                             "accepted": sum(r["accepted"] for r in refine_log), "components": refine_log}
            last = last + "2" if last == "s1" else "s2"
            save_variant(dst / last, masks)
        if c3["enabled"]:
            masks, add_log = add_similar(proc, img, masks, probs, no_obj, c3, cfg["damage_phrases"], obj_frac, wins)
            log["stage3"] = {"added_instances": len(add_log), "instances": add_log}
            last = last + "3" if last in ("s1", "s12", "s2") else "s3"
            save_variant(dst / last, masks)

    coco = export_coco(masks, probs, Path(info["image"]).name, h, w, cfg.get("coco_min_area_px", 16))
    (dst / "predictions.coco.json").write_text(json.dumps(coco))
    cv2.imwrite(str(dst / "overlay.jpg"), cv2.cvtColor(overlay(img, masks, obj, info.get("class_colors_rgb", {})),
                                                       cv2.COLOR_RGB2BGR))
    log["final_variant"], log["seconds"] = last, round(time.time() - t0, 1)
    (dst / "sam3_info.json").write_text(json.dumps(log, indent=1))
    return log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_root", required=True, help="folder with one MVP_clean output folder per image")
    p.add_argument("--images_dir", required=True, help="folder with the original images")
    p.add_argument("--config", required=True, help="e.g. ../configs/sam3_pp.yaml")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="re-process folders that already have sam3_pp/")
    p.add_argument("--max_gpu_fraction", type=float, default=None, help="optional hard cap on GPU memory")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path.insert(0, os.getcwd())  # SAM3_LoRA root
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.max_gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction)
    need_inst = bool(cfg["stage2_refine"]["enabled"])
    model = build_sam3_image_model(bpe_path=cfg["sam3"]["bpe_path"], device="cuda", eval_mode=True,
                                   load_from_HF=True, enable_inst_interactivity=need_inst)
    proc = Sam3Processor(model)
    predictor = model if need_inst else None  # stage 2 calls model.predict_inst (needs enable_inst_interactivity)

    dirs = sorted(d for d in Path(args.pred_root).iterdir() if (d / "run_info.json").exists())
    if not args.overwrite:
        dirs = [d for d in dirs if not (d / "sam3_pp" / "sam3_info.json").exists()]
    dirs = dirs[: args.max_images] if args.max_images else dirs
    print(f"[sam3_pp] {len(dirs)} image folders to process in {args.pred_root}")
    failed = []
    for k, d in enumerate(dirs, start=1):
        try:
            log = process_image(d, args, cfg, proc, predictor)
            s1 = log.get("stage1", {}).get("removed", {})
            print(f"[sam3_pp] ({k}/{len(dirs)}) {d.name}: {log['seconds']}s, windows={log['windows']}, "
                  f"stage1 removed={sum(v['components_removed'] for v in s1.values())} comps, "
                  f"stage2 accepted={log.get('stage2', {}).get('accepted', '-')}/{log.get('stage2', {}).get('components_tried', '-')}, "
                  f"stage3 added={log.get('stage3', {}).get('added_instances', '-')}", flush=True)
        except Exception:
            print(f"[sam3_pp] ({k}/{len(dirs)}) FAILED {d.name}:\n{traceback.format_exc()}", flush=True)
            failed.append(d.name)
        torch.cuda.empty_cache()
    if failed:
        (Path(args.pred_root) / "sam3_pp_failed.txt").write_text("\n".join(failed) + "\n")
        print(f"[sam3_pp] {len(failed)} failed -> {Path(args.pred_root) / 'sam3_pp_failed.txt'}")


if __name__ == "__main__":
    main()
