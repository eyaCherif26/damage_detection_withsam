"""Run one of MVP_clean's scripts (train / evaluate / infer_orthophoto / visualize_predictions / ...) with the
Qwen3-VL backbone `qwen3vl_semantic` registered - without changing any MVP_clean file.

MVP_clean's location is an argument (the project may have another name on other machines). Run from the MVP_clean
folder, as MVP_clean's own scripts expect (relative paths like runs/ and results/ are resolved from there):

  cd <MVP_clean folder>
  python <VLM_test>/scripts/run_mvp_clean.py --mvp_clean_dir . \
      --backbone_config <VLM_test>/configs/qwen3vl/backbone_lora.yaml \
      --script train -- --config <VLM_test>/configs/qwen3vl/qwen3vl_cif_512.yaml

Everything after `--` is passed unchanged to MVP_clean's script. For `--script train`, the backbone YAML is copied
next to the run's checkpoints (<checkpoint_dir>/<name>/backbone_config.yaml); evaluate/infer must be given the
SAME backbone YAML that was used for training.
"""
from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

import yaml

VLM_TEST_ROOT = Path(__file__).resolve().parent.parent


def _arg_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main() -> None:
    if "--" not in sys.argv:
        raise SystemExit("usage: run_mvp_clean.py --mvp_clean_dir DIR --backbone_config YAML --script NAME -- <script args>")
    split = sys.argv.index("--")
    own, rest = sys.argv[1:split], sys.argv[split + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mvp_clean_dir", required=True, type=Path, help="root of the MVP_clean project (any folder name)")
    p.add_argument("--backbone_config", required=True, type=Path, help="Qwen3-VL backbone settings YAML")
    p.add_argument("--script", required=True, help="MVP_clean script name in <mvp_clean_dir>/scripts/, without .py")
    args = p.parse_args(own)

    mvp = args.mvp_clean_dir.resolve()
    script = mvp / "scripts" / f"{args.script}.py"
    if not script.exists():
        raise SystemExit(f"MVP_clean script not found: {script}")
    # MVP_clean first: VLM_test also has folders named configs/ and data/, which must not shadow MVP_clean's
    sys.path.insert(0, str(VLM_TEST_ROOT))
    sys.path.insert(0, str(mvp))
    import models  # noqa: F401  (MVP_clean: registers its built-in backbones)
    from qwen_seg.qwen3vl_semantic import register

    settings = yaml.safe_load(args.backbone_config.read_text()) or {}
    register(settings)
    print(f"[run_mvp_clean] registered qwen3vl_semantic with {args.backbone_config}: {settings}")

    if args.script == "train" and (cfg_path := _arg_value(rest, "--config")):  # provenance
        from configs.loader import load_experiment_config

        cfg = load_experiment_config(cfg_path)
        ckpt_dir = Path(_arg_value(rest, "--checkpoint_dir") or cfg.training.checkpoint_dir) / cfg.name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.backbone_config, ckpt_dir / "backbone_config.yaml")

    sys.argv = [str(script)] + rest
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
