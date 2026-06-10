# Artifact Packaging

Tài liệu này mô tả cách repo submission đóng gói bằng chứng artifact mà không biến repo thành kho chứa checkpoint hoặc dataset research.

## Checkpoint final của Module 1

- Checkpoint tham chiếu mạnh nhất cho Module 1 là `R013_REPRO`.
- Source-of-truth bên ngoài repo:

```text
F:\deeplearning\experiment_value\module1_retrain_sequence\R013_REPRO\best_iou.ckpt
```

- Local workspace hiện có checkpoint Module 1 ở path:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

- Binary này bị ignore theo policy và không nên commit.

## Vì sao checkpoint binary bị ignore

- Giữ repo nhẹ và dễ nộp.
- Tách source code khỏi weights.
- Tránh kéo thêm artifact research không cần thiết vào Git history.

## Dataset là external

Các dataset training chính không nằm trong Git:

- `old_photo_pairs_10_hq`
- `r013_finetune_set`

Repo chỉ nên giữ:

- manifest dữ liệu;
- sample/golden nhỏ;
- docs giải thích provenance;
- script verify artifact local.

## Reproduction run manifests

Manifest trong `artifacts/manifests/` mô tả:

- checkpoint lineage;
- reproduction runs;
- dataset provenance;
- artifact nào local, artifact nào external, artifact nào chỉ historical.

## Verify command

```bash
python -B scripts/verify_artifacts.py check-all --repo-root .
```

Có thể kiểm tra riêng:

```bash
python -B scripts/verify_artifacts.py check-checkpoints --repo-root .
python -B scripts/verify_artifacts.py check-datasets --repo-root .
```

## Boundary

Đây là artifact evidence packaging cho submission repo.

- Không phải automatic training pipeline.
- Không tự tải checkpoint.
- Không tự copy dataset.
- Không thay thế repo research gốc.
