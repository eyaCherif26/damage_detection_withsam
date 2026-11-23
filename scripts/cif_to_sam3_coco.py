"""Convert a COCO split (any dataset) into the SAM3_LoRA layout: <out>/<split>/_annotations.coco.json + image symlinks.

Categories are remapped with a label-map YAML (native name -> target name, null = drop), then renamed to
descriptive text prompts (--prompt_name Target=prompt). Images without (remaining) annotations are kept as
negatives. All paths are arguments; nothing is specific to one project layout.

Example (CIF, 40-image smoke subset):
  python scripts/cif_to_sam3_coco.py --split train \
      --coco .../cif_coco/train/annotations.json --images .../cif_coco/train/images \
      --label_map .../MVP_clean/configs/data/label_maps/cif6_to_mvp5.yaml \
      --out data/cif_sam3_smoke --max_images 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_PROMPTS = {
    "Cracks": "crack",
    "Wetspot": "wet spot",
    "Spalling": "spalling",
    "Rust": "rust",
    "Efflorescence": "efflorescence",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", required=True, help="source COCO json")
    p.add_argument("--images", required=True, help="directory holding the source images")
    p.add_argument("--label_map", required=True, help="YAML: native category name -> target name (null = drop)")
    p.add_argument("--out", required=True, help="output root (SAM3_LoRA training.data_dir)")
    p.add_argument("--split", required=True, help="output split folder name: train / valid / test")
    p.add_argument("--prompt_name", nargs="*", default=[], help="Target=prompt overrides, e.g. Cracks=crack")
    p.add_argument("--max_images", type=int, default=None, help="random subset size (seeded)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prompts = dict(DEFAULT_PROMPTS)
    for kv in args.prompt_name:
        k, v = kv.split("=", 1)
        prompts[k] = v

    label_map = yaml.safe_load(Path(args.label_map).read_text())
    coco = json.loads(Path(args.coco).read_text())
    images_dir = Path(args.images).resolve()

    native = {c["id"]: c["name"] for c in coco["categories"]}
    unmapped = sorted(set(native.values()) - set(label_map))
    if unmapped:
        raise SystemExit(f"Categories missing from label map {args.label_map}: {unmapped}")
    targets = sorted({t for t in label_map.values() if t is not None} & {label_map[n] for n in native.values()},
                     key=lambda t: list(prompts).index(t) if t in prompts else 99)
    missing_prompt = [t for t in targets if t not in prompts]
    if missing_prompt:
        raise SystemExit(f"No prompt name for target(s) {missing_prompt}; pass --prompt_name Target=prompt")
    new_id = {t: i + 1 for i, t in enumerate(targets)}
    cat_map = {cid: new_id[label_map[name]] for cid, name in native.items() if label_map[name] is not None}

    images = coco["images"]
    if args.max_images is not None and args.max_images < len(images):
        images = random.Random(args.seed).sample(images, args.max_images)
    keep_ids = {im["id"] for im in images}

    anns, src_counts = [], Counter()
    for a in coco["annotations"]:
        if a["image_id"] not in keep_ids:
            continue
        src_counts[native[a["category_id"]]] += 1
        if a["category_id"] in cat_map:
            anns.append({**a, "category_id": cat_map[a["category_id"]]})

    split_dir = Path(args.out) / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        link = split_dir / im["file_name"]
        if not link.exists():
            src = images_dir / im["file_name"]
            if not src.exists():
                raise SystemExit(f"Image not found: {src}")
            os.symlink(src, link)

    out = {
        "images": images,
        "annotations": anns,
        # name = text prompt used by SAM 3; mvp_class = MVP_clean class name, used for metric/W&B labels
        "categories": [{"id": new_id[t], "name": prompts[t], "mvp_class": t, "supercategory": "damage"} for t in targets],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(out))

    tgt_counts = Counter(prompts[t] for a in anns for t in targets if new_id[t] == a["category_id"])
    with_ann = len({a["image_id"] for a in anns})
    print(f"[{args.split}] {len(images)} images ({with_ann} annotated, {len(images) - with_ann} negatives), "
          f"{len(anns)} annotations -> {split_dir}")
    print(f"  source counts: {dict(sorted(src_counts.items()))}")
    print(f"  target counts: {dict(tgt_counts)}")


if __name__ == "__main__":
    main()
