# External Dependencies

## LaMa

The project repository does not vendor the official LaMa source tree. The pipeline calls LaMa through an external runtime wrapper.

- This is an official/pretrained integration.
- The repository does not claim LaMa fine-tuning.
- `configs/external_paths.example.yaml` is only a template.
- `configs/external_paths.yaml` is machine-specific and should not be committed.

## Checkpoint Policy

- Segmentation checkpoints are not committed to Git by default.
- The external artifact reference for `R013_REPRO` is configured through a local artifact root, for example:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- SHA256:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`

## CodeFormer

- CodeFormer is an optional dependency.
- Module 3 should not be described as a completed end-to-end capability.
- If a qualitative demo is shown in another context, it should still be described only as qualitative or prototype evidence.
- The repository does not claim identity preservation.

## Why External Dependencies Are Not Committed

- To keep the project repository lightweight and reviewable.
- To avoid hard-coded local runtime paths in the source tree.
- To use `configs/external_paths.example.yaml` plus the manifests under `artifacts/manifests/` for local path mapping.
- To separate source code from weights, runtimes, and research artifacts.
