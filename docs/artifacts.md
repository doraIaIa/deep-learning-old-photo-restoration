# Artifact Packaging

> **Tóm tắt tiếng Việt**
> 
> - Tài liệu này giải thích cách đóng gói minh chứng đầu ra (artifact packaging).
> - Mô tả lý do tại sao dataset và checkpoint weights không được commit trực tiếp vào Git.
> - Các phép đo mở rộng trong tương lai chỉ mang tính chất tham khảo, không đại diện cho kết quả hiện tại.

This document explains how the repository packages artifact evidence without turning the source tree into a storage location for checkpoints, research datasets, or large runtime outputs.

## Module 1 checkpoint lineage

- The strongest operational checkpoint reference for Module 1 is `R013_REPRO`.
- The source of truth lives outside Git and is typically mapped through a local artifact root, for example:

```text
<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt
```

- A local workspace can map that checkpoint into the public-facing skeleton:

```text
checkpoints/segmenter/r013_final/
```

- The checkpoint binary remains a local ignored artifact and should not be committed.
- Dataset lineage and artifact paths are documented under `artifacts/manifests/` and can be mapped locally through `configs/external_paths.example.yaml`.
- The public checkpoint skeleton is documented in `checkpoints/segmenter/r013_final/` and `checkpoints/segmenter/r009_synthetic_pretrain/`. Those folders do not contain checkpoint binaries.

## Synthetic pretraining data lineage

The synthetic pretraining lineage for Module 1 is documented as manifest-driven provenance:

1. Clean images from DIV2K
2. Crack-source imagery and annotations from CrackForest
3. A processed crack bank stored as RGBA assets
4. The synthetic dataset `ds-crack3d-512-n1000-v001`
5. Historical synthetic runs `R006` through `R009`

`R009` is the synthetic pretraining or initialization stage. It should not be described as the final real-domain checkpoint.

## Real-domain fine-tuning progression

After synthetic initialization:

- `R010_REPRO`: fine-tuning on `old_photo_pairs_10_hq` with thin masks
- `R011_REPRO`: fine-tuning on repair-mask targets
- `R012_REPRO`: an experimental manual-mask branch
- `R013_REPRO`: the final controlled Module 1 reproduction on `r013_finetune_set`

## Dataset and checkpoint policy

- Full datasets are external and are not committed to Git.
- Checkpoint binaries are local ignored artifacts and are not committed to Git.
- The repository keeps manifests, split metadata, small documentation assets, and minimal examples only.

## Manifest-driven verification

The manifests under `artifacts/manifests/` and `data/manifests/` describe:

- dataset provenance
- synthetic lineage
- reproduction run lineage
- checkpoint availability
- whether an artifact is external, historical, or local-only

You can validate the manifests with:

```bash
python -B scripts/verify_artifacts.py check-checkpoints --repo-root .
python -B scripts/verify_artifacts.py check-datasets --repo-root .
python -B scripts/verify_artifacts.py check-all --repo-root .
```

## Boundary

- The repository does not auto-download checkpoints.
- The repository does not auto-copy datasets.
- The repository does not package full automatic training reproduction.
- The repository does not claim LaMa fine-tuning.
- LPIPS/FID are not claimed for this artifact.

### Segmenter Checkpoints (R006-R013)
The segmenter checkpoints follow canonical experiment-run names (e.g., seg-unet-attn-r013-gen120-fixed118-local) to precisely map to historical logs and metrics.
- **R006-R008**: These are skeleton-only folders because the binary weights are missing from the audit, preserving lineage evidence.
- **R009-R013**: These are available locally if copied from <LOCAL_ARTIFACT_ROOT>.
- **Current Alias**: checkpoints/segmenter/current/ is an alias conceptually pointing to the selected R013 canonical folder without duplicating binaries.
- **Binary Policy**: All checkpoint binaries (.ckpt, .pth) are strictly **local ignored** and **not committed** to the repository to maintain claim-safety and small repository size.
