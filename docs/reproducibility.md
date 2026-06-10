# Reproducibility

## Current Reproducible Scope

This repository currently targets minimal reproducibility for:

- Module 1 segmentation using the `R013_REPRO` checkpoint reference;
- the pipeline `segmentation -> hybrid mask -> official/pretrained LaMa`;
- the `demo3` golden regression case.

## Module 1 Facts That Must Stay Accurate

- `R013` starts from `120` images but only `118` valid image-mask pairs exist.
- The missing `masks_fixed` entries are `real_0099` and `real_0112`.
- The fixed split is `83 / 18 / 17`.
- `R013_REPRO` initializes from `R011_REPRO`.
- The main threshold for reporting and fair comparison is `0.50`.

## External Artifacts Required

- `configs/external_paths.yaml`
- an external segmentation checkpoint
- the official LaMa repository and weights
- the optional CodeFormer repository and weights if the user wants to explore that branch

External checkpoint reference:

```text
<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt
```

SHA256:

```text
5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203
```

## Minimal Replay Path for Demo3

### 1. Prepare external paths

- Copy `configs/external_paths.example.yaml` to `configs/external_paths.yaml`.
- Fill in the local paths for the LaMa runtime and checkpoint artifacts.

### 2. Run readiness checks

```bash
python scripts/check_readiness.py
```

### 3. Replay the auto-mask smoke case for `demo3`

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --output-dir examples/outputs/seg_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png ^
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Expected outputs:

- `final_mask.png`
- `restored_before_face.png`
- `metadata.json`

### 4. Replay the mask-bypass smoke case for `demo3`

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --mask examples/golden/demo3_r013_repair_wide/final_mask.png ^
  --output-dir examples/outputs/pipeline_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

Expected smoke/regression behavior:

- output size matches the golden reference;
- `metadata.json` is produced;
- current smoke artifacts show `MAE = 0` and `PSNR = inf` for the mask-bypass path.

## Evaluation Boundary

- `demo3` can be reused for smoke/regression checks.
- README and docs should not be interpreted as evidence that LPIPS/FID or full end-to-end evaluation has been completed.
- The strongest quantitative evidence currently remains the Module 1 segmentation metrics.

## Known Boundaries

- This document describes a minimal smoke/regression replay path, not a full reproduction protocol for the entire research history.
- `R013_REPRO` is the strongest claim-safe checkpoint reference, but the checkpoint binary itself is not committed to Git by default.
