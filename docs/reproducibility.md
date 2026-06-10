# Reproducibility

## Reproducible Scope Hiện Tại

Repo submission hiện nhắm tới reproducibility tối thiểu cho:

- Module 1 segmentation theo checkpoint tham chiếu `R013_REPRO`;
- pipeline `segmentation -> hybrid mask -> official/pretrained LaMa`;
- golden regression case `demo3`.

## Module 1 Facts Cần Giữ Đúng

- `R013` có `120` ảnh ban đầu nhưng chỉ `118` valid image-mask pairs.
- Hai ảnh thiếu `masks_fixed`: `real_0099`, `real_0112`.
- Split cố định: `83 / 18 / 17`.
- `R013_REPRO` init từ `R011_REPRO`.
- Threshold chính cho báo cáo/fair comparison: `0.50`.

## External Artifacts Required

- `configs/external_paths.yaml`
- checkpoint segmentation ngoài repo
- official LaMa repo và weight
- optional CodeFormer repo và weight nếu người dùng muốn tự thử nhánh optional

Checkpoint tham chiếu ngoài repo:

```text
F:\deeplearning\experiment_value\module1_retrain_sequence\R013_REPRO\best_iou.ckpt
```

SHA256:

```text
5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203
```

## Minimal Replay Path For Demo3

### 1. Chuẩn bị external paths

- copy `configs/external_paths.example.yaml` thành `configs/external_paths.yaml`
- điền path local cho LaMa runtime và checkpoint

### 2. Kiểm tra readiness

```bash
python scripts/check_readiness.py
```

### 3. Replay auto-mask smoke cho `demo3`

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

### 4. Replay mask-bypass smoke cho `demo3`

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --mask examples/golden/demo3_r013_repair_wide/final_mask.png ^
  --output-dir examples/outputs/pipeline_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

Expected behavior ở mức smoke/regression:

- output cùng kích thước với golden reference
- có `metadata.json`
- với smoke artifact hiện có, mask-bypass path cho thấy `MAE = 0` và `PSNR = inf`

## Evaluation Boundary

- Có thể tái sử dụng `demo3` cho smoke/regression.
- Không nên dùng README/docs của repo submission để suy ra rằng LPIPS/FID hoặc full end-to-end evaluation đã hoàn tất.
- Các metric có bằng chứng mạnh nhất hiện tại vẫn là segmentation metrics của Module 1.

## Known Boundaries

- Tài liệu này mô tả replay path tối thiểu cho smoke/regression, không phải protocol tái lập đầy đủ toàn bộ lịch sử thực nghiệm.
- `R013_REPRO` là checkpoint tham chiếu claim safety mạnh nhất hiện tại, nhưng repo submission không commit checkpoint đó vào Git theo mặc định.
