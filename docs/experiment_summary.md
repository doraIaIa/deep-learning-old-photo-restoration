# Experiment Summary

This project repository does not bundle the full research history. This document keeps the points that matter for claim safety and minimal reproducibility.

## Module 1 Summary

- The strongest reproduced checkpoint reference is `R013_REPRO`.
- `R013` starts from `120` images but only `118` valid image-mask pairs exist in `masks_fixed`.
- The missing `masks_fixed` entries are `real_0099` and `real_0112`.
- The fixed split is `83 / 18 / 17`.
- `R013_REPRO` initializes from `R011_REPRO`, not from `R012_REPRO`.
- The main reporting threshold is `0.50`.
- Historical `R013` validation IoU is `0.381231`; reproduced validation IoU is close at `0.380532`.
- Fair test `R013_REPRO @0.50`: IoU/F1 = `0.337970 / 0.501339`.
- Fair test `R011_REPRO @0.55`: IoU/F1 = `0.246848 / 0.394876`.
- Delta fair comparison: `+0.091122 IoU`, `+0.106463 F1`.

## R012 Status

- `R012` is an experimental manual-subset branch with `15` samples.
- `R012` is not the final improvement.
- `R012` is not used as the initialization checkpoint for `R013`.

## Module 2 Summary

- The repository currently uses official/pretrained LaMa through an external runtime wrapper.
- It does not claim LaMa fine-tuning.
- Losses such as `L1`, `perceptual`, and `adversarial` should be treated as future work unless fine-tuning artifacts are available.

## Evaluation Boundary

- Strong evidence exists for Module 1 segmentation metrics such as `IoU`, `F1`, `Precision`, and `Recall`.
- The repository does not claim completed `LPIPS`, `FID`, or `masked-region LPIPS`.
- `demo3` is a golden regression case for smoke/demo checks, not a full benchmark over the real old-photo set.

## Operational Evidence Kept in the Repository

- The repository keeps smoke/golden artifacts for `demo3` under `seg_smoke_demo3`, `pipeline_smoke_demo3`, and `gradio_smoke_demo3`.
- These artifacts are useful for operational checks and regression inspection.
- They should not be interpreted as full end-to-end quantitative evaluation.

## Minimal Evidence Kept in the Repository

- README and docs describing the current operating scope.
- Small golden artifacts for `demo3`.
- CLI and Gradio demo support for smoke/readiness checks.

## Moved to Future Work

- LaMa fine-tuning.
- LPIPS/FID/masked-region LPIPS evaluation.
- Full end-to-end quantitative evaluation.
- A complete Module 3 face restoration flow.
