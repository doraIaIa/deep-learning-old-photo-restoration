# Khả năng Tái tạo (Reproducibility)

## Phạm vi Tái tạo Hiện tại (Current Reproducible Scope)

Repo hiện tại cung cấp khả năng tái tạo tối thiểu (minimal reproducibility) cho:

- Module 1 segmentation sử dụng tham chiếu checkpoint `R013_REPRO`;
- Pipeline: `segmentation -> hybrid mask -> official/pretrained LaMa`;
- Golden regression case `demo3`.

## Các thông tin về Module 1 cần được giữ nguyên (Module 1 Facts That Must Stay Accurate)

- `R013` bắt đầu từ `120` ảnh nhưng chỉ có `118` valid image-mask pairs.
- Các mục `masks_fixed` bị thiếu là `real_0099` và `real_0112`.
- Tập chia cố định (fixed split) là `83 / 18 / 17`.
- `R013_REPRO` được khởi tạo từ `R011_REPRO`.
- Ngưỡng chính (main threshold) cho báo cáo và fair comparison là `0.50`.

## Các External Artifacts cần có (External Artifacts Required)

- `configs/external_paths.yaml`
- Một external segmentation checkpoint
- Repo official LaMa và weights
- Repo CodeFormer tùy chọn và weights nếu người dùng muốn chạy thử.

Tham chiếu external checkpoint (External checkpoint reference):

```text
<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt
```

SHA256:

```text
5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203
```

## Đường dẫn tái tạo tối thiểu cho Demo3 (Minimal Replay Path for Demo3)

### 1. Chuẩn bị external paths

- Sao chép `configs/external_paths.example.yaml` thành `configs/external_paths.yaml`.
- Điền các đường dẫn cục bộ cho LaMa runtime và các checkpoint artifacts.

### 2. Chạy readiness checks

```bash
python scripts/check_readiness.py
```

### 3. Replay quá trình auto-mask smoke case cho `demo3`

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --output-dir examples/outputs/seg_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png ^
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Các output dự kiến (Expected outputs):

- `final_mask.png`
- `restored_before_face.png`
- `metadata.json`

### 4. Replay quá trình mask-bypass smoke case cho `demo3`

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --mask examples/golden/demo3_r013_repair_wide/final_mask.png ^
  --output-dir examples/outputs/pipeline_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

Hành vi smoke/regression dự kiến (Expected smoke/regression behavior):

- Kích thước ảnh đầu ra khớp với golden reference;
- Sinh ra file `metadata.json`;
- Các smoke artifacts hiện tại hiển thị `MAE = 0` và `PSNR = inf` đối với đường dẫn mask-bypass.

## Giới hạn Đánh giá (Evaluation Boundary)

- Có thể tái sử dụng `demo3` cho smoke/regression checks.
- README và tài liệu không nên được diễn giải là minh chứng rằng LPIPS/FID hay full end-to-end evaluation đã được hoàn thiện.
- Minh chứng định lượng mạnh nhất hiện tại vẫn nằm ở các Module 1 segmentation metrics.

## Ranh giới đã biết (Known Boundaries)

- Tài liệu này chỉ mô tả minimal smoke/regression replay path, không phải toàn bộ quy trình tái tạo cho cả lịch sử nghiên cứu.
- `R013_REPRO` là tham chiếu checkpoint an toàn nhất (strongest claim-safe checkpoint reference), nhưng bản thân checkpoint binary mặc định sẽ không được commit vào Git.



## Lưu ý về Bố cục Artifact (Artifact Layout Note)
- `examples/golden/` là file tham chiếu chuẩn (frozen expected reference).
- `examples/outputs/` được tạo ra khi chạy script cục bộ (local examples).
- `outputs/` là thư mục root được khuyên dùng để tạo thí nghiệm mới ở máy local.
