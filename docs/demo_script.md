# Demo Script

## Presentation Goal

- Present the project repository as a modular pipeline operating within the scope `Module 1 + hybrid mask + official/pretrained LaMa`.
- Avoid describing unfinished components as if they were production-ready capabilities.

## Suggested 3-5 Minute Flow

1. State the current reproducible scope clearly:
   `R013_REPRO` is the operational checkpoint reference for Module 1.
2. State the `R013` dataset facts clearly:
   the initial set had `120` images but only `118` valid image-mask pairs exist in `masks_fixed`.
3. Emphasize that the main threshold used for reporting and fair comparison is `0.50`.
4. Introduce the core pipeline:
   segmentation -> hybrid mask -> `repair_wide_v1` -> official/pretrained LaMa.
5. State that Module 3 is currently optional/prototype and not part of the required flow.

## Safe Talking Points

- Why a modular pipeline:
  separating segmentation, mask refinement, and inpainting improves observability and avoids overclaiming a complete end-to-end model when the strongest evidence currently sits in Module 1.
- Why a hybrid mask:
  the learned mask and the heuristic mask compensate for each other before refinement.
- Why checkpoints are not committed to Git:
  the project repository keeps code, templates, and documentation; checkpoints are referenced through external paths or manifests.
- Why LaMa fine-tuning is not claimed:
  the repository currently uses official/pretrained LaMa through an external runtime wrapper.
- Why LPIPS/FID are not claimed:
  those metrics do not yet have complete artifacts in the current repository.

## Things Not to Say

- Do not say `R013` has `120 valid pairs`.
- Do not say `R012` is the final improvement.
- Do not say LaMa was fine-tuned.
- Do not say CodeFormer preserves identity.
- Do not say the repository already has full quantitative end-to-end evaluation.
