# Limitations

## Current Scope

The current pipeline focuses on:

- segmentation for damaged regions;
- hybrid mask construction;
- `repair_wide_v1`;
- an official/pretrained LaMa wrapper.

## What Is Not a Safe Claim Yet

- LaMa fine-tuning.
- `LPIPS`, `FID`, and `masked-region LPIPS`.
- Full quantitative end-to-end evaluation.
- CodeFormer identity preservation.
- A complete Module 3 face restoration flow.
- Illumination handling as a completed implementation in the current repository.

## Dataset and Experiment Caveats

- `R013` must always be described as `120` initial images but only `118` valid pairs.
- `R012` is only an experimental branch with `15` manual samples.
- `demo3` is a golden regression case, not a benchmark representing the full real old-photo set.

## Future Work

- LaMa fine-tuning with complete artifacts.
- LPIPS/FID/masked-region LPIPS.
- A stronger end-to-end Module 3 flow.
- A more complete evaluation protocol beyond smoke/regression checks.
