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
- LPIPS/FID/masked-region LPIPS evaluation (planned for future work, not currently claimed).
- Full end-to-end quantitative evaluation.
- A complete Module 3 face restoration flow.


## Training Lineage: R006-R013 Segmenter Development

| Run | Objective | Data / Label Target | Key Result | Decision |
|---|---|---|---|---|
| **R006** | Baseline (synthetic) | Synthetic (50 ep) | Val IoU: 0.3852, Val F1: 0.5249 (thr 0.25) | Recall weak; move to augmentation |
| **R007** | Strong augmentation | Synthetic aug | Val IoU: 0.3912, Val F1: 0.5257 (thr 0.20) | Precision improved; recall weak; change loss |
| **R008** | BCE + Tversky loss | Synthetic aug | Val IoU: 0.4064, Val F1: 0.5492 (thr 0.70) | Recall improved; extend training |
| **R009** | Synthetic pretrain (60 ep) | Synthetic aug | Val IoU: 0.4171, Val F1: 0.5595. **Real Test IoU: 0.0022** | Severe domain gap on real photos; use as base |
| **R010** | Real-domain fine-tune | Real (thin masks) | Real Test IoU: 0.2927, Test F1: 0.4528 (thr 0.70) | Domain gap overcome; masks too thin for LaMa |
| **R011** | Repair mask fine-tune | Real (repair masks) | Test IoU: 0.4478, Test F1: 0.6186 | Stable baseline; missed extremely thin cracks |
| **R012** | Manual mask constraint | Manual (15 samples) | Test IoU: 0.2846, Test F1: 0.4430 | Overfit/small-data negative experiment |
| **R013** | Operational segmenter | Fixed 118 pairs | Val F1: 0.5502, Test IoU: 0.3456 (thr 0.50) | Selected operational checkpoint (seg-unet-attn-r013-gen120-fixed118-local) |

## Failure-Driven Design Decisions

- **Modular Pipeline**: Earlier direct end-to-end restoration attempts motivated the decomposition into a modular segmentation and inpainting pipeline to mitigate regression bias.
- **Inpainting Dependency**: LaMa is utilized strictly as a pretrained wrapper subprocess. No fine-tuning is performed on LaMa to avoid unstable generative training sequences.

## Training Data Evolution

- **Initial Datasets Rejected**: Datasets similar to CrackForest or ds-crack3d-512-n0200-v001 were rejected due to mask area mismatch (e.g., asphalt cracks are thicker than photo scratches).
- **Synthetic Pretraining Data**: Adopted Crack Bank RGBA assets combined with physically grounded 3D degradation, normal maps, Phong illumination, and alpha blending over DIV2K backgrounds.
- **Domain Gap & Fine-tuning**: The severe domain gap observed in R009 (IoU dropping to 0.0022 on real test) necessitated a real-domain fine-tuning sequence (R010, R011, R013) using curated real photographs.


## Mask and Threshold Strategy

- **Loss Progression**: Initially started with BCE+Dice. To heavily penalize false negatives, R008 introduced Tversky Loss (alpha=0.3, beta=0.7). R011 increased beta=0.8 for recall-oriented repair masks.
- **Threshold Evolution**: Inference thresholds varied dynamically based on model confidence distribution (e.g., R007 at 0.20, R009 at 0.90). The final R013 segmenter uses a stable operational threshold of 0.50.
- **Hybrid Mask Refinement**: The deep learning mask is unioned with a classical CV branch (CLAHE, Blackhat, Canny). The 
epair_wide_v1 strategy then applies morphological closing, connection, and dilation to prepare the final mask for inpainting.

