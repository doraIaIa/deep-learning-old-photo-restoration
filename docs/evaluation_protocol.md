# Evaluation Protocol

> **Tóm tắt tiếng Việt**
> 
> - Tài liệu này mô tả giao thức đánh giá (evaluation protocol) hiện tại.
> - Phạm vi chứng minh hiện tại tập trung vào module phân vùng (segmentation) và kiểm thử smoke/regression với `demo3`.
> - Các phép đo không thuộc scope hiện tại sẽ được đánh dấu là chưa claim (not claimed).

## Current Scope

The strongest evaluation evidence currently available in the repository is for Module 1 segmentation and for smoke or regression behavior around `demo3`.
This document describes the intended protocol and the evidence that is currently present. It should not be interpreted as proof that every evaluation runner has been fully packaged.

## Safe Metrics

- `IoU`
- `F1`
- `Precision`
- `Recall`

These are the metrics with the strongest evidence across the `R010_REPRO -> R013_REPRO` lineage, with `R013_REPRO` serving as the strongest current checkpoint reference.

## Safe Wording

- The main threshold used for reporting and fair comparison for `R013` is `0.50`.
- `R013` should be described as `120` initial images with only `118` valid pairs.
- `demo3` is a golden regression case for smoke and demo validation.

## Claims That Should Remain Deferred

- `LPIPS` (not claimed)
- `FID` (not claimed)
- `masked-region LPIPS` (not claimed)
- full quantitative end-to-end evaluation

Those items should be described as planned evaluation or future work unless matching artifacts are clearly available in the repository.

## Runner Caveat

- Segmentation metrics are strongly evidenced by the audited artifacts and summaries.
- The repository does not yet package the full runner stack for these protocols, oracle-mask protocols, or full ablation coverage.
- `scripts/evaluate_segmentation.py` and `scripts/run_ablation.py` currently provide protocol coverage and summaries for available artifacts rather than a full experimental orchestration layer.
- For that reason, the current docs should not be interpreted as evidence that the entire evaluation stack is complete.

