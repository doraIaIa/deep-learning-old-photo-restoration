<div align="center">

# Old Photo Restoration with Deep Learning

### Modular Restoration Pipeline

**Crack Segmentation · Hybrid Mask Construction · Official LaMa Wrapper**

</div>

---

## Introduction

This project repository packages a modular pipeline for old photo restoration. The current operational scope focuses on:

- detecting crack-like or small tear regions with segmentation;
- combining learned masks with classical heuristic masks;
- sending the final mask to official/pretrained LaMa through an external runtime wrapper.

This repository does not attempt to retell the entire research history. Claims in this README are limited to what is strongly evidenced in `experiment_value` and in the current codebase.

## Current Reproducible Scope

- The strongest operational checkpoint reference is `R013_REPRO`.
- The current inference chain is `segmentation -> hybrid mask -> repair_wide_v1 -> official/pretrained LaMa`.
- The repository provides a CLI pipeline via `scripts/run_pipeline.py`, a readiness check via `scripts/check_readiness.py`, and a local demo via `scripts/run_gradio_demo.py`.
- Module 3 face restoration is not part of the main operational flow.
- `demo3` is a golden smoke/regression case for demonstration and sanity checking, not a full benchmark over the real old-photo set.

## What Is Implemented

- A U-Net style segmenter with Attention Gate for Module 1.
- A hybrid mask made from `dl mask + cv mask + repair_wide_v1`.
- An external wrapper for official/pretrained LaMa.
- A main pipeline CLI at `scripts/run_pipeline.py`.
- A local Gradio demo at `scripts/run_gradio_demo.py`.
- Readiness checks for dependencies, configuration, and checkpoint paths.
- Small golden artifacts for `demo3` to support smoke/regression checks.

## Module 1 Safe Claims

- The final operational/reproducible checkpoint reference is `R013_REPRO`.
- The `R013` dataset started from `120` images, but only `118` valid image-mask pairs exist in `masks_fixed`.
- The missing `masks_fixed` entries are `real_0099` and `real_0112`.
- The fixed split used in reproduction summaries is `83 / 18 / 17` for `train / val / test`.
- `R013_REPRO` initializes from `R011_REPRO`, not from `R012_REPRO`.
- The main threshold for reporting and fair comparison is `0.50`.
- The strongest evidenced Module 1 metrics are `IoU`, `F1`, `Precision`, and `Recall`.

## What Is Experimental

- `R012` is an experimental branch with `15` manual samples.
- `R012` does not outperform `R011` convincingly and is not used as the initialization checkpoint for `R013`.
- Notes about very low thresholds or highly sensitive modes should be treated as optional inference modes, not as the main metric claim.
- Some evaluation and ablation material is still minimal and should not be interpreted as a complete evaluation stack.

## What Is Future Work

- LaMa fine-tuning with complete training artifacts.
- LPIPS, FID, and masked-region LPIPS. (Future evaluation protocol / not claimed)
- Full end-to-end quantitative evaluation for the entire pipeline.
- A complete Module 3 face restoration flow.
- Identity-preservation metrics for face restoration.
- Colorization, super-resolution, ONNX/TensorRT export, and tiling for high-resolution images.

## Pipeline Overview

```text
Input image
  -> Module 1 segmentation
  -> CV mask support
  -> union mask
  -> repair_wide_v1 refinement
  -> official/pretrained LaMa wrapper
  -> restored output
```

The pipeline prioritizes intermediate observability so that segmentation, mask refinement, and inpainting errors can be isolated more easily.

## Evaluation Boundary

- The strongest quantitative evidence in this repository is for Module 1 segmentation metrics.
- `demo3` is a golden regression case for pipeline behavior and smoke/demo checks.
- This repository does not claim completed `LPIPS`, `FID`, `masked-region LPIPS`, or full quantitative end-to-end evaluation.
- This repository does not claim perfect old photo restoration.

## Checkpoint and Artifact Policy

- Checkpoints are not committed to Git by default.
- The repository prefers `external checkpoint path + manifest + SHA256`.
- The external checkpoint reference for `R013_REPRO` is configured through a local artifact root, for example:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- The SHA256 of the `R013_REPRO` checkpoint is:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`
- Local machine-specific configuration such as `configs/external_paths.yaml` should not be committed. Use `configs/external_paths.example.yaml` and the manifests under `artifacts/manifests/` to map local paths.

## LaMa and Module 3 Caveats

- The repository uses official/pretrained LaMa through an external runtime wrapper.
- It does not claim that LaMa was fine-tuned in this repository.
- Losses such as `L1`, `perceptual`, and `adversarial` should be treated as design directions or future work unless fine-tuning artifacts are available.
- CodeFormer should currently be described only as optional, prototype, or future work depending on context.
- The repository does not claim CodeFormer identity preservation.

## Quickstart

### 1. Create an environment

```bash
python -m venv .venv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a local config

```bash
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

### 4. Run readiness checks

```bash
python scripts/check_readiness.py
```

### 5. Run the local demo

```bash
python scripts/run_gradio_demo.py
```

### 6. Run the minimal auto-mask CLI example

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --output-dir examples/outputs/seg_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png ^
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

The CLI uses these defaults:

- `configs/inference.yaml`
- `configs/checkpoints.yaml`
- `configs/external_paths.yaml`

If external dependencies are not ready, stop at `check_readiness.py` and review `docs/demo_script.md` and `docs/reproducibility.md`.

## Expected Outputs

Typical outputs in the current scope include:

- a predicted segmentation mask;
- a refined/hybrid mask after `repair_wide_v1`;
- a restored image before the optional face module (`restored_before_face.png`);
- a `metadata.json` file with runtime configuration, backend, threshold, and smoke/regression statistics when available.

Current smoke artifacts include output layouts such as:

- `examples/outputs/seg_smoke_demo3/`
- `examples/outputs/pipeline_smoke_demo3/`
- `examples/outputs/gradio_smoke_demo3/`

## Repository Layout

```text
app/
configs/
docs/
examples/
scripts/
src/old_photo_restoration/
```

## Safe Claims and Limitations

- This repository is strongest on Module 1 segmentation and hybrid masking.
- Module 2 is currently a pretrained LaMa wrapper, not a LaMa fine-tuning implementation.
- Module 3 is not yet a complete operational flow.
- Demo, smoke, and golden artifacts support minimal reproducibility and regression checking; they are not a replacement for a large benchmark.
- `R013` should always be described as `120` initial images but only `118` valid image-mask pairs.
- `LPIPS`, `FID`, and `masked-region LPIPS` are not complete artifacts in the current repository.
- Module 3 does not yet have quantitative identity-preservation evaluation.

Detailed claim-safety notes from the earlier documentation pass remain available in `docs/scope_and_claim_safety.md`.

