"""Score MVP_clean predictions and their SAM 3 post-processed variants against COCO ground truth (CPU only).

Variants (each image is scored only if ALL requested variants exist for it, so the comparison is fair):
  base         MVP_clean prob >= its thresholds, from <image>/prob_<class>.png + run_info.json (needs no SAM 3)
  s1 s12 s123  <image>/sam3_pp/<variant>/mask_<class>.png from VLM_test/scripts/sam3_postprocess.py
  name=path    any predictions COCO file inside each image folder, e.g. hyst=hyst/predictions.coco.json from
               MVP_clean's postprocess_predictions.py (run it with --segmentation_format rle --min_area 0)
Metrics (MVP_clean definitions, via sam3_eval.PixelMetrics), inside the no-data validity mask:
  pixel level per class: IoU, Dice (= F1), precision, recall; macro, macro_present, frequency_weighted
  image level presence per class: is the class in the image / did we predict it -> precision, recall, F1,
  plus exact-match ratio (all evaluated classes right) per image.
Evaluated classes = model classes that the label map can produce (e.g. infra_hafen: Cracks, Spalling,
Efflorescence only), so classes the dataset never annotates are not counted as false positives.

    python scripts/score_predictions.py --pred_root <mvp outputs> --coco <gt annotations.json> \
        --label_map <native->MVP yaml> --images_dir <images> --out <dir> [--variants base s1 s12 s123]
Outputs in --out: metrics.csv (pixel), presence.csv, stage_counts.csv, metrics.json; a summary is printed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from sam3_eval import PixelMetrics, _gt_union, no_data_validity

AGG = ("macro", "macro_present", "frequency_weighted")
KEYS = ("iou", "dice", "precision", "recall")


def coco_variant_masks(path: Path, classes: list[str], h: int, w: int) -> dict | None:
    """Masks from a predictions COCO file whose category names are the MVP classes (e.g. MVP_clean's
    postprocess_predictions.py output). Use segmentation_format rle there: polygon rings lose holes."""
    if not path.exists():
        return None
    pred = json.loads(path.read_text())
    name_of = {c["id"]: c["name"] for c in pred["categories"]}
    return {c: _gt_union([a for a in pred["annotations"] if name_of.get(a["category_id"]) == c], h, w)
            for c in classes}


def variant_masks(d: Path, variant: str, classes: list[str], thresholds: dict) -> dict | None:
    if "=" in variant:  # name=relative/path/predictions.coco.json
        p0 = cv2.imread(str(d / f"prob_{classes[0]}.png"), cv2.IMREAD_GRAYSCALE)
        if p0 is None:
            return None
        return coco_variant_masks(d / variant.split("=", 1)[1], classes, *p0.shape)
    out = {}
    for c in classes:
        if variant == "base":
            p = cv2.imread(str(d / f"prob_{c}.png"), cv2.IMREAD_GRAYSCALE)
            if p is None:
                return None
            out[c] = p.astype(np.float32) / 255.0 >= thresholds[c]
        else:
            m = cv2.imread(str(d / "sam3_pp" / variant / f"mask_{c}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            out[c] = m > 127
    return out


class Presence:
    def __init__(self, classes: list[str]) -> None:
        self.classes = classes
        self.n = {k: np.zeros(len(classes)) for k in ("tp", "fp", "fn", "tn")}
        self.images = self.exact = 0

    def update(self, preds: dict, gts: dict) -> None:
        ok = True
        for i, c in enumerate(self.classes):
            p, g = bool(preds[c].any()), bool(gts[c].any())
            self.n["tp" if p and g else "fp" if p else "fn" if g else "tn"][i] += 1
            ok &= p == g
        self.images += 1
        self.exact += ok

    def compute(self) -> dict:
        tp, fp, fn = self.n["tp"], self.n["fp"], self.n["fn"]
        prec = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full_like(tp, np.nan), where=(2 * tp + fp + fn) > 0)
        per = {c: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "tp": int(tp[i]), "fp": int(fp[i]),
                   "fn": int(fn[i]), "tn": int(self.n["tn"][i])} for i, c in enumerate(self.classes)}
        return {"per_class": per, "exact_match_ratio": self.exact / max(1, self.images), "images": self.images}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_root", required=True)
    p.add_argument("--coco", required=True, help="ground-truth COCO annotations.json")
    p.add_argument("--label_map", required=True, help="YAML: native category name -> MVP class (null = ignore)")
    p.add_argument("--images_dir", required=True, help="original images (for the no-data validity mask)")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=["base", "s1", "s12", "s123"])
    p.add_argument("--classes", nargs="+", default=None, help="override the evaluated classes")
    p.add_argument("--no_validity", action="store_true", help="do not exclude black no-data borders")
    args = p.parse_args()

    label_map = yaml.safe_load(Path(args.label_map).read_text())
    coco = json.loads(Path(args.coco).read_text())
    cat_target = {c["id"]: label_map.get(c["name"]) for c in coco["categories"]}
    unmapped = sorted(c["name"] for c in coco["categories"] if c["name"] not in label_map)
    if unmapped:
        print(f"[score] WARNING: GT categories not in the label map (ignored): {unmapped}")
    gt_image = {Path(im["file_name"]).stem: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    dirs = sorted(d for d in Path(args.pred_root).iterdir() if (d / "run_info.json").exists())
    pix, pres, counts = {}, {}, defaultdict(lambda: defaultdict(int))
    classes = None
    skipped = defaultdict(list)
    for d in dirs:
        info = json.loads((d / "run_info.json").read_text())
        stem = Path(info["image"]).stem
        if stem not in gt_image:
            skipped["no ground truth"].append(d.name)
            continue
        thresholds = info["thresholds"]
        if classes is None:
            targets = {t for t in label_map.values() if t}
            classes = args.classes or [c for c in thresholds if c in targets]
            pix = {v: PixelMetrics(classes) for v in args.variants}
            pres = {v: Presence(classes) for v in args.variants}
            print(f"[score] evaluated classes: {classes}")
        preds = {v: variant_masks(d, v, classes, thresholds) for v in args.variants}
        missing = [v for v, m in preds.items() if m is None]
        if missing:
            skipped[f"missing variant(s) {missing}"].append(d.name)
            continue
        im = gt_image[stem]
        h, w = preds[args.variants[0]][classes[0]].shape
        if (im["height"], im["width"]) != (h, w):
            skipped["size mismatch with GT"].append(d.name)
            continue
        if args.no_validity:
            valid = np.ones((h, w), dtype=bool)
        else:
            bgr = cv2.imread(str(Path(args.images_dir) / Path(info["image"]).name), cv2.IMREAD_COLOR)
            if bgr is None:
                skipped["image not found in --images_dir"].append(d.name)
                continue
            valid = no_data_validity(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        gts = {c: _gt_union([a for a in anns[im["id"]] if cat_target.get(a["category_id"]) == c], h, w) & valid
               for c in classes}
        for v in args.variants:
            for i, c in enumerate(classes):
                pix[v].update(i, preds[v][c], gts[c], valid)
            pres[v].update({c: preds[v][c] & valid for c in classes}, gts)
        info_path = d / "sam3_pp" / "sam3_info.json"
        if info_path.exists():
            si = json.loads(info_path.read_text())
            counts["images"]["n"] += 1
            for c, r in si.get("stage1", {}).get("removed", {}).items():
                counts["stage1_components_removed"][c] += r["components_removed"]
            for r in si.get("stage2", {}).get("components", []):
                counts["stage2_tried"][r["class"]] += 1
                counts["stage2_accepted"][r["class"]] += int(r["accepted"])
            for r in si.get("stage3", {}).get("instances", []):
                counts["stage3_added"][r["class"]] += 1

    for reason, names in skipped.items():
        print(f"[score] skipped {len(names)} folder(s): {reason} (e.g. {names[:3]})")
    if classes is None or not pres or pres[args.variants[0]].images == 0:
        raise SystemExit("[score] nothing scored")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = {v: v.split("=", 1)[0] for v in args.variants}  # "hyst=hyst/predictions.coco.json" -> "hyst"
    result = {label[v]: {"pixel": pix[v].compute(), "presence": pres[v].compute()} for v in args.variants}
    args.variants = [label[v] for v in args.variants]
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    with open(out / "metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "class", *KEYS, "gt_pixels"])
        for v in args.variants:
            m = result[v]["pixel"]
            for c, r in m["per_class"].items():
                wr.writerow([v, c, *[f"{r[k]:.4f}" for k in KEYS], r["gt_pixels"]])
            for a in AGG:
                wr.writerow([v, a, *[f"{m[a][k]:.4f}" for k in KEYS], ""])
    with open(out / "presence.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "class", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])
        for v in args.variants:
            pr = result[v]["presence"]
            for c, r in pr["per_class"].items():
                wr.writerow([v, c, *[f"{r[k]:.4f}" for k in ("precision", "recall", "f1")],
                             *[r[k] for k in ("tp", "fp", "fn", "tn")]])
            wr.writerow([v, "exact_match_ratio", f"{pr['exact_match_ratio']:.4f}", "", "", "", "", "", ""])
    with open(out / "stage_counts.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["count", *classes])
        for k, per in counts.items():
            if k != "images":
                wr.writerow([k, *[per.get(c, 0) for c in classes]])

    n = result[args.variants[0]]["presence"]["images"]
    print(f"\n[score] {n} images, classes {classes}")
    print(f"{'variant':8s} {'dice':>7s} {'iou':>7s} {'prec':>7s} {'recall':>7s}  (macro_present)  "
          + "  ".join(f"{c[:10]:>10s}" for c in classes) + "  (dice per class)   presence-F1   exact-match")
    for v in args.variants:
        m, pr = result[v]["pixel"], result[v]["presence"]
        mp = m["macro_present"]
        pf1 = np.nanmean([r["f1"] for r in pr["per_class"].values()])
        print(f"{v:8s} {mp['dice']:7.4f} {mp['iou']:7.4f} {mp['precision']:7.4f} {mp['recall']:7.4f}                  "
              + "  ".join(f"{m['per_class'][c]['dice']:10.4f}" for c in classes)
              + f"   {pf1:11.4f}   {pr['exact_match_ratio']:11.4f}")
    if counts:
        print(f"\nstage counts over {counts['images']['n']} images -> {out / 'stage_counts.csv'}")
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
