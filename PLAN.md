# 2026-11-24: Qwen3-VL-8B vision encoder as a semantic-segmentation backbone (MVP_clean used as a library)
Decisions: only the vision tower (27 layers, 1152 ch, 16-px patches; mean/std 0.5) + fpn_upernet_lite decoder +
LoRA; code in VLM_test, MVP_clean unchanged (path = --mvp_clean_dir, the project has another name on the remote);
runs: CIF balanced subset -> dacl10k -> combined, same settings as segformer_cif_512.
- qwen_seg/qwen3vl_semantic.py: downloads only the shard with `model.visual.` weights (model-00004-of-00004 for 8B),
  Qwen-processor patch layout, OFFICIAL Qwen3VLVisionModel forward + hooks on feature_layers [8,16,24,26],
  ImageNet->Qwen renormalisation inside, LoRA (qkv, proj), register() into MVP_clean's registry.
- scripts/run_mvp_clean.py: registers the backbone, runs any MVP_clean script (train/evaluate/infer_orthophoto/...).
- configs/qwen3vl/: backbone_lora.yaml, qwen3vl_{cif,dacl,cif_dacl}_512.yaml, datasets/cif_coco_balanced.yaml
  (placeholder paths, set by the user).
- tests/test_qwen3vl_semantic.py: tiny random config, CPU. Not run yet.
# 2026-11-24: SAM 3 orthophoto inference in MVP_clean's output format
- New scripts/sam3_infer_orthophoto.py: overlapping 1008-px tiles (linear-ramp blending), per-class heat maps
  (sam3_eval.class_heatmap), no-data -> 0, outputs prob_<class>.png / predictions.coco.json (rle|polygon) /
  overlay.png (+preview) / run_info.json exactly like MVP_clean infer_orthophoto.py -> MVP_clean post-processing
  and sam3_postprocess.py run on it unchanged. Options as MVP_clean: --scale, --crop, --overlap, --min_area,
  --thresholds (thresholds_best.json from heat-map validation). Not run yet.

# 2026-11-23: LoRA on the DETR encoder (image-text fusion) on 24 GB, class-balanced 1k CIF subset
- Memory fix = opt-in efficient attention (lora_layers use_sdpa via `optimizations.efficient_attention`) + optional
  bf16; SAM 3 already checkpoints activations in the DETR encoder and the ViT (use_act_checkpoint=True).
- configs/cif_detr_enc_smoke.yaml (40 imgs, 1 epoch) then configs/cif_detr_enc_1k.yaml (15 epochs).
- cif_to_sam3_coco.py --balanced_per_class K (opt-in): rarest class first, each class in >= K images.
- Compare with cif_24gb (decoder-only) using heat-map eval on the SAME validation/test images.

# 2026-11-23: lost changes re-applied (files had reverted to the ~2026-11-18 state during the git steps)
Pushed as 3 separate branches / pull requests: (1) Full-LoRA optimizations (lora_layers use_sdpa, trainer
`optimizations` block, configs/cif_full_24gb.yaml), (2) heat-map evaluation (sam3_eval.py --mode heatmap),
(3) post-processing (sam3_postprocess.py: predict_inst fix, near-miss scores, look-alike rules, sam3_filtered/;
sam3_pp.yaml object list + shadow rules + stages 2/3 off; score_predictions.py COCO variants).
Experiment YAML paths are managed by the user - never changed by Claude.
Next (user request): test-set visualisation + threshold diagnostics (presence score) in heat-map mode.
- [x] 2026-11-23 sam3_eval.py heat-map mode: --heat_vis N (image+GT, per-class heat map with GT outline, max heat,
  presence) and --diagnostics (diagnostics.csv + per-class presence/score summary). Opt-in, not run yet.
  Hypothesis to test: weak Spalling/Rust = presence head too low after fine-tuning (instance score = query x presence).

# SAM3 "Full" LoRA on 24 GB (2026-11-18) - opt-in, existing configs unchanged
- New configs/cif_full_24gb.yaml: LoRA on all components except text encoder (as repo full_lora_config), rank 32,
  batch 1, num_negatives 0, lr 5e-5, 10 epochs, data cif_sam3_1k (1k train / 300 valid), W&B run sam3_lora_cif_full_1k.
- New opt-in `optimizations:` block (absent in all older configs -> original code path):
  efficient_attention (lora_layers.MultiheadAttentionLoRA.use_sdpa -> F.scaled_dot_product_attention) and
  bf16_autocast (trainer forward in bf16, outputs cast back to fp32). Repo's `mixed_precision` key intentionally ignored.
- SAM 3 ViT activation checkpointing is already on by default in training (vitdet.py use_act_checkpoint=True).
- Not run yet: SDPA-vs-original equivalence check (CPU), smoke run (40 imgs, 1 epoch) for peak memory, then 1k run.

# Active plan (2026-11-17): SAM 3 (zero-shot) as post-processing for MVP_clean predictions

Decisions (user, 2026-11-17): base model `segformer_native_cif_512` (MVP_clean/runs/segformer_native_cif_512/best.pt,
thresholds results/segformer_native_cif_512/thresholds.json); test on CIF test + the 89-image infra_hafen set;
damage ON non-structural objects is removed; object prompts: ladder, fender, chain, rope, bollard, pipe, railing,
cable, boat, buoy, sign, vegetation, person, artifacts; zero-shot SAM 3 (no LoRA) for now.
No changes to MVP_clean. Every command is proposed to the user first; SAM 3 GPU work runs on the 24 GB machine.

## Data flow (file in -> file out, like MVP_clean's postprocess_predictions.py / ssl_false_alarm_filter.py)
1. MVP_clean (mvp_clean_env): `scripts/infer_orthophoto_batch.sh` per image -> `<root>/<image>/prob_<class>.png`,
   `run_info.json` (thresholds). No MVP_clean post-processing (raw baseline).
2. SAM 3 (sam3_lora_env, 24 GB machine): new `VLM_test/scripts/sam3_postprocess.py` reads each `<root>/<image>/`
   + the image, writes `<root>/<image>/sam3_pp/`: mask_<class>.png per variant, predictions.coco.json,
   sam3_info.json (every decision logged), overlay.png.
3. Scoring (CPU, any env): new `VLM_test/scripts/score_predictions.py`: per-class precision/recall/Dice(F1)/IoU,
   macro, presence, per variant (base / +S1 / +S1+S2 / +S1+S2+S3), using sam3_eval.py's PixelMetrics +
   no-data rule (MVP_clean definitions). Baseline cross-checked against MVP_clean evaluate.py --coco --sliding_window.

## Step 0 - verify (read-only, before coding)
- Location + label map of infra_hafen COCO GT (images: new_data_infracloud/hafen_v11_annotationen/bildern; existing
  outputs viz/segformer_native_cif_512/infra_hafen/ki_results may be reusable if raw).
- CIF test size/resolution; dev subset (e.g. 100 imgs) vs full test.
- SAM 3 APIs: text prompts (Sam3Processor.set_text_prompt), exemplar boxes (add_geometric_prompt), and the
  interactive box/point predictor (enable_inst_interactivity / inst_interactive_predictor) for Stage 2.

### Step 0 findings (2026-11-17, read-only)
- infra_hafen label map MVP_clean/configs/data/label_maps/infra_hafen_to_mvp5.yaml: only Spalling, Crack->Cracks,
  Efflorescence -> report macro_present. COCO GT file path: still to get from user.
- MVP_clean step runs locally (user): `python scripts/infer_orthophoto.py --config configs/experiments/segformer_native_cif_512.yaml
  --checkpoint runs/segformer_native_cif_512/best.pt --image <img> --output_dir <dir>` (or infer_orthophoto_batch.sh);
  without --thresholds it uses results/segformer_native_cif_512/thresholds.json. Writes prob_<class>.png (8-bit),
  run_info.json (thresholds, image path), predictions.coco.json, overlay.png. Baseline mask = prob/255 >= threshold.
- SAM 3 APIs (verified in code): Sam3Processor.set_text_prompt (text), add_geometric_prompt(box cxcywh normalized,
  label pos/neg; with or without text) -> Stage 1 + Stage 3; build_sam3_image_model(enable_inst_interactivity=True)
  -> model.inst_interactive_predictor = SAM3InteractiveImagePredictor (SAM2 API: set_image(np RGB),
  predict(point_coords Nx2 px, point_labels, box xyxy px, mask_input 1x256x256 logits, multimask_output)
  -> masks CxHxW, quality scores) -> Stage 2 (class-agnostic; class comes from MVP_clean; MVP_clean prob can be
  passed as mask_input). To verify on first run: sam3.pt loads the interactive (tracker) weights without missing keys.

### Code written (2026-11-17, not run yet)
- configs/sam3_pp.yaml (object prompts, damage phrases, all stage thresholds)
- scripts/sam3_postprocess.py (stages 1-3, writes <image>/sam3_pp/{base,s1,s12,s123}/mask_<class>.png, objects.png,
  predictions.coco.json, overlay.jpg, sam3_info.json)
- scripts/score_predictions.py (CPU; variants base/s1/s12/s123; pixel + presence metrics; stage counts)
- infra_hafen GT: MVP_clean/data/processed/hafen_v11_annotationen_merged/train/annotations.json,
  images: new_data_infracloud/hafen_v11_annotationen/bildern

### Chosen workflow (user, 2026-11-18): inference -> SAM 3 stage 1 only -> MVP_clean postprocess_predictions.py
`--hysteresis_low_threshold Cracks=0.1 Spalling=0.1 Efflorescence=0.6 --morph_classes Cracks --morph_close_px 2
--morph_open_px 1 --merge_classes Cracks --merge_gap_px 30`. Stages 2/3 off. Test sets: infra_hafen, CIF test
(balanced test annotations.json), Seehafen2_test. Compare base / s1 / pp alone / s1+pp.

### Earlier test order (2026-11-18): infra_hafen, inference -> SAM 3 filter (stage 1) -> hysteresis
- sam3_postprocess.py stage 1 also writes <image>/sam3_filtered/ (prob maps zeroed on objects + removed components,
  run_info.json) -> MVP_clean postprocess_predictions.py --input_dir <image>/sam3_filtered --output_dir <image>/sam3_hyst.
- Stages 2/3 disabled in configs/sam3_pp.yaml for this test (they output masks, not probabilities).
- Score: base, s1, (hyst alone), sam3_hyst with score_predictions.py (name=path COCO variants supported).

## Stage 1 - remove damage on objects
- SAM 3 text queries for the object list -> union object mask (score >= obj_threshold, dilate a few px).
  Large images: tile (~1008 px, overlap) so small objects (chains, ropes) are not lost.
- Per class, per connected component of the MVP_clean mask: remove the component if >= overlap_frac (0.5) lies on
  objects; remove remaining damage pixels inside object masks anyway (damage on objects is out of scope).
- "artifacts" is vague for SAM 3: test it, alternatives e.g. "black border", "blurred area", "stitching seam".

## Stage 2 - refine remaining damage masks
- Per component: crop with margin; SAM 3 prompt = class phrase + component box + top-k MVP_clean-confident points
  (interactive predictor; fallback: text+exemplar box in the crop, take the instance with best IoU).
- Accept only if IoU(new, original) >= 0.3 and mean MVP_clean prob inside new mask >= low threshold; else keep
  original. Cracks: watch CIF region-style labels vs SAM thin lines (score per class).

## Stage 3 - find missed similar damage
- Per image and class with confirmed components: SAM 3 text + confirmed components as positive exemplar boxes.
- New instance = IoU with existing mask < 0.1; accept if mean MVP_clean prob >= weak threshold (hysteresis idea),
  area >= min_area, not on Stage-1 objects.

## Honest evaluation
- Parameters (thresholds, overlap_frac, IoU gates) tuned on CIF val / a calibration half of infra_hafen, never on
  the reported test images. Report all 4 variants side by side + per-stage counts (removed/refined/added).

# Earlier plan: SAM ViT-B + LoRA semantic backbone in MVP_clean (2026-11-17)
(SAM 3 plan paused until facebook/sam3 access is granted -> see PLAN_sam3_lora_paused.md)

Task spec: `Prompt - SAM LoRA Semantic CIF.md`. Test on this machine (RTX 3050 8 GB) that the workflow trains;
full training later on RTX PRO 4000 Blackwell (24 GB, sm_120; mvp_clean_env torch 2.8.0+cu128 is compatible).

## Phase 0 - verify (read-only)
- transformers `SamVisionModel`: module names (qkv/proj), pos_embed shape, rel_pos sizes, global-attn block ids, image-size check.
- `facebook/sam-vit-base` in HF cache? (public, ungated, ~375 MB download otherwise).
- Trainer: backbone_parameters()/head_parameters() + frozen_epochs freezing -> in LoRA mode backbone_parameters() must be LoRA-only.
- Where the transform sets ImageNet mean/std; existing `--max_train_images` subset flag.

## Implementation (minimal, existing patterns)
1. `models/sam_semantic.py` registered `sam_semantic`: SAM image encoder at 512 (pos_embed + global rel_pos interpolated),
   custom LoRALinear on qkv (opt. proj), modes lora / frozen / full_finetune, decoder fpn_upernet_lite (ViTDet simple
   pyramid from global-attn blocks + FPN) + optional hi-res CNN branch, gradient checkpointing.
2. `configs/schema.py` additive: ModelConfig.normalization, hires_branch, decoder accepts fpn_upernet_lite; LoraConfig;
   TrainingConfig.gradient_checkpointing. Defaults = existing behaviour.
3. Normalization hook: SAM mean/std when model.normalization == "sam" (documented deviation).
4. Configs (configs/experiments/): sam_vitb_lora_cif_512.yaml (8 GB), sam_vitb_lora_cif_512_24gb.yaml, ablations nolora / nohires.
5. tests/test_sam_semantic.py. 6. docs/methodology_sam_lora.md, CLAUDE.md + README entries.

## Verification (local)
pytest (new + full suite); VRAM/speed table batch 1/2/4/8; smoke train 64 imgs x 2 epochs (wandb offline);
evaluate.py + infer_orthophoto.py on the checkpoint; --resume check. Then handover section for the 24 GB machine.

## Progress
- [x] (2026-11-17) sam_semantic model + 4 configs + tests/test_sam_semantic.py (11) done; full MVP_clean suite 465 passed. scripts/measure_vram.py added. Not yet run: VRAM table, smoke train.
- [x] HF login done; official facebook/sam3 sam3.pt downloaded, sha256 9999e234...8c9e verified (identical to the Tao re-upload). VLM_test/scripts/sam3_zero_shot.py written, not yet run.
- [x] (2026-11-17) SAM3 LoRA runs on the 24 GB machine: OOM fixed with apply_to_detr_encoder: false (repo's naive attention, issue #39) + batch 1; Triton needs gcc on that machine; validate_sam3_lora.py crashes on non-0..N-1 COCO image ids (fix pending: renumber ids in converter).
- [x] (2026-11-17) Evaluation + W&B like MVP_clean: new VLM_test/scripts/sam3_eval.py (per-class IoU/Dice=F1/precision/recall, macro/macro_present/freq-weighted, no-data exclusion, COCO predictions, panels; CLI for zero-shot or LoRA); patch in SAM3_LoRA/train_sam3_lora_native.py (per-epoch metrics on val + train subset, MVP_clean W&B keys/table/images, best_lora_weights.pt by val_macro_dice, best weights as W&B artifact); configs gained best_metric/eval/wandb blocks (project damage_detection). Not run yet.
- [!] Machine/WSL crashed while a model was running (cause unknown, not from a run started by Claude). Next GPU runs: cap memory, one image, bf16, monitor nvidia-smi.
