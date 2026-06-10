# Artifact Packaging

Tài liệu này mô tả cách repo đóng gói bằng chứng artifact mà không biến repository thành kho chứa checkpoint, dataset research hoặc runtime outputs lớn.

## Module 1 checkpoint lineage

- Checkpoint tham chiếu mạnh nhất cho Module 1 là `R013_REPRO`.
- Source-of-truth ngoài repo được cấu hình qua local artifact path ngoài Git; xem thêm:

```text
<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt
```

- Workspace local hiện có checkpoint Module 1 tại:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

- Binary checkpoint vẫn là local ignored artifact và không nên commit.
- Dataset lineage và artifact paths chi tiết được ghi trong `artifacts/manifests/` và được ánh xạ cục bộ qua `configs/external_paths.example.yaml`.
- Public-facing checkpoint skeleton được documented tại `checkpoints/segmenter/r013_final/` và `checkpoints/segmenter/r009_synthetic_pretrain/`; các skeleton này không chứa binary.

## Synthetic pretraining data lineage

Synthetic pretraining lineage cho Module 1 được mô tả bằng manifest-driven provenance:

1. Clean images từ DIV2K
2. Crack-source imagery và annotations từ CrackForest
3. Processed crack bank dưới dạng RGBA assets
4. Synthetic dataset `ds-crack3d-512-n1000-v001`
5. Historical synthetic runs `R006` đến `R009`

`R009` giữ vai trò synthetic pretraining/init stage. Đây không phải checkpoint real-domain cuối cùng.

## Real-domain fine-tuning progression

Sau synthetic initialization:

- `R010_REPRO`: fine-tune trên `old_photo_pairs_10_hq` với thin masks
- `R011_REPRO`: fine-tune trên repair-mask target
- `R012_REPRO`: nhánh manual-mask mang tính experimental
- `R013_REPRO`: final controlled reproduction cho Module 1 trên `r013_finetune_set`

## Dataset và checkpoint policy

- Full datasets là external và không commit vào Git.
- Checkpoint binaries là local ignored artifacts và không commit vào Git.
- Repo chỉ giữ manifests, split metadata, docs assets nhỏ và sample tối thiểu.

## Manifest-driven verification

Manifest trong `artifacts/manifests/` và `data/manifests/` mô tả:

- dataset provenance
- synthetic lineage
- reproduction run lineage
- checkpoint availability
- artifact nào external, historical hoặc local-only

Kiểm tra manifest:

```bash
python -B scripts/verify_artifacts.py check-checkpoints --repo-root .
python -B scripts/verify_artifacts.py check-datasets --repo-root .
python -B scripts/verify_artifacts.py check-all --repo-root .
```

## Boundary

- Repo này không tự tải checkpoint.
- Repo này không tự copy dataset.
- Repo này không đóng gói full training reproduction tự động.
- Repo này không claim LaMa fine-tune.
- Repo này không claim LPIPS/FID hoàn chỉnh nếu artifact tương ứng chưa được đóng gói.
