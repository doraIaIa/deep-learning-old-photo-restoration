<div align="center">

# Phục Hồi Ảnh Cũ Với Deep Learning

### Old Photo Restoration — Modular Deep Learning Pipeline

**Crack Segmentation · Hybrid Mask Construction · Official LaMa Wrapper**

</div>

---

## Giới thiệu

Repo submission này đóng gói một pipeline theo hướng mô-đun cho bài toán phục hồi ảnh cũ. Trọng tâm hiện tại là:

- phát hiện vùng hư hại dạng vết nứt hoặc rách nhỏ bằng segmentation;
- hợp nhất mask học sâu với mask heuristic cổ điển;
- đưa mask cuối sang official/pretrained LaMa thông qua wrapper chạy external runtime.

Repo này không cố mô tả toàn bộ lịch sử thực nghiệm nghiên cứu trước đó. Các claim trong README được giới hạn theo những gì đang có bằng chứng mạnh trong `experiment_value` và trong code hiện tại của repo submission.

## Current Reproducible Scope

- Checkpoint vận hành có bằng chứng tái lập mạnh nhất hiện tại là `R013_REPRO`.
- Chuỗi vận hành hiện tại của submission là `segmentation -> hybrid mask -> repair_wide_v1 -> official/pretrained LaMa`.
- Repo hiện có CLI pipeline qua `scripts/run_pipeline.py`, readiness check qua `scripts/check_readiness.py`, và local demo qua `scripts/run_gradio_demo.py`.
- Module 3 face restoration chưa phải thành phần vận hành chính trong repo submission hiện tại.
- `demo3` là golden case phục vụ smoke/regression và trình diễn, không phải benchmark đầy đủ trên toàn bộ tập ảnh cũ thực.

## What Is Implemented

- Segmenter kiểu U-Net có Attention Gate cho Module 1.
- Hybrid mask gồm `dl mask + cv mask + repair_wide_v1`.
- Wrapper gọi official/pretrained LaMa qua external runtime hoặc subprocess.
- CLI pipeline chính qua `scripts/run_pipeline.py`.
- Local Gradio demo qua `scripts/run_gradio_demo.py`.
- Readiness check cho dependency, config và checkpoint path.
- Golden artifacts nhỏ cho `demo3` để phục vụ smoke/regression.

## Module 1 Safe Claims

- Final operational/reproducible checkpoint đang được tham chiếu an toàn là `R013_REPRO`.
- Bộ dữ liệu `R013` xuất phát từ `120` ảnh, nhưng chỉ có `118` cặp ảnh-mask hợp lệ trong `masks_fixed`.
- Hai ảnh thiếu `masks_fixed` là `real_0099` và `real_0112`.
- Split cố định dùng trong tóm tắt tái lập là `83 / 18 / 17` cho `train / val / test`.
- `R013_REPRO` khởi tạo từ `R011_REPRO`, không khởi tạo từ `R012_REPRO`.
- Threshold chính để trình bày và fair comparison là `0.50`.
- Các metric có bằng chứng mạnh nhất hiện tại cho Module 1 là `IoU`, `F1`, `Precision`, `Recall`.

## What Is Experimental

- `R012` chỉ là một nhánh thực nghiệm với `15` manual samples.
- `R012` không vượt `R011` một cách thuyết phục và không được dùng làm init cho `R013`.
- Các ghi chú về threshold rất thấp hoặc chế độ “sensitive” chỉ nên xem là mode suy luận tùy chọn, không phải claim metric chính.
- Một phần tài liệu evaluation và ablation trong repo submission vẫn ở trạng thái tối thiểu và sẽ được hoàn thiện khi có thêm artifact tương ứng.

## What Is Future Work

- Fine-tune LaMa với artifact huấn luyện đầy đủ.
- LPIPS, FID và masked-region LPIPS.
- Đánh giá định lượng end-to-end đầy đủ cho toàn pipeline.
- Module 3 face restoration hoàn chỉnh trong repo submission.
- Identity-preservation metrics cho face restoration.
- Colorization, super-resolution, ONNX/TensorRT, tiling cho ảnh độ phân giải cao.

## Pipeline Tổng Quan

```text
Input image
  -> Module 1 segmentation
  -> CV mask support
  -> union mask
  -> repair_wide_v1 refinement
  -> official/pretrained LaMa wrapper
  -> restored output
```

Pipeline ưu tiên khả năng quan sát trung gian, giúp tách riêng lỗi ở segmentation, mask refinement và inpainting.

## Evaluation Boundary

- Repo submission hiện có bằng chứng mạnh nhất cho segmentation metrics của Module 1.
- `demo3` chỉ là golden regression case để kiểm tra hành vi pipeline và smoke/demo.
- Không claim rằng repo hiện đã hoàn tất `LPIPS`, `FID`, `masked-region LPIPS` hoặc full quantitative end-to-end evaluation.
- Không claim “khôi phục hoàn hảo ảnh cũ”.

## Checkpoint And Artifact Policy

- Checkpoint không commit vào Git theo mặc định.
- Repo ưu tiên `external checkpoint path + manifest + SHA256`.
- Checkpoint tham chiếu ngoài repo cho `R013_REPRO` được cấu hình qua local artifact root, ví dụ:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- SHA256 của checkpoint `R013_REPRO`:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`
- Config local như `configs/external_paths.yaml` vẫn là cấu hình theo máy và không nên commit. Dùng `configs/external_paths.example.yaml` và các manifest trong `artifacts/manifests/` để map local paths.

## LaMa And Module 3 Caveats

- Repo submission dùng official/pretrained LaMa qua wrapper external runtime.
- Không claim LaMa đã được fine-tune trong repo này.
- Các loss như `L1`, `perceptual`, `adversarial` chỉ nên xem là hướng thiết kế hoặc future work nếu chưa có artifact fine-tune rõ ràng.
- CodeFormer hiện chỉ nên được mô tả là optional, prototype hoặc future work tùy ngữ cảnh.
- Không claim CodeFormer bảo toàn danh tính.

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
```

### 2. Cài dependency

```bash
pip install -r requirements.txt
```

### 3. Tạo config local

```bash
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

### 4. Kiểm tra readiness

```bash
python scripts/check_readiness.py
```

### 5. Chạy local demo

```bash
python scripts/run_gradio_demo.py
```

### 6. Chạy CLI auto-mask tối thiểu

Ví dụ replay path hiện có cho `demo3`:

```bash
python scripts/run_pipeline.py ^
  --image examples/inputs/demo3.png ^
  --output-dir examples/outputs/seg_smoke_demo3 ^
  --face-mode off ^
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png ^
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

CLI trên dùng mặc định:

- `configs/inference.yaml`
- `configs/checkpoints.yaml`
- `configs/external_paths.yaml`

Nếu external dependency chưa sẵn sàng, nên dừng ở bước `check_readiness.py` và xem thêm `docs/demo_script.md`, `docs/reproducibility.md`.

## Expected Outputs

Khi chạy CLI/demo ở scope hiện tại, output chính thường gồm:

- predicted mask từ segmentation;
- refined/hybrid mask sau `repair_wide_v1`;
- restored image trước optional face module (`restored_before_face.png`);
- `metadata.json` ghi cấu hình chạy, backend, threshold và các thống kê smoke/regression nếu có.

Trong các smoke artifacts đang có, có thể thấy các đường dẫn output kiểu:

- `examples/outputs/seg_smoke_demo3/`
- `examples/outputs/pipeline_smoke_demo3/`
- `examples/outputs/gradio_smoke_demo3/`

## Repository Layout

```text
app/
configs/
docs/
examples/
scripts/
src/old_photo_restoration/
```

## Safe Claims And Limitations

- Repo này mạnh nhất ở Module 1 segmentation và hybrid mask.
- Module 2 hiện là pretrained LaMa wrapper, không phải LaMa fine-tune.
- Module 3 chưa phải phần hoàn chỉnh của submission flow.
- Demo, smoke và golden artifacts hiện phục vụ reproducibility tối thiểu và regression, không thay cho benchmark lớn.
- `R013` phải được hiểu là `120` ảnh ban đầu nhưng chỉ `118` valid image-mask pairs.
- `LPIPS`, `FID` và `masked-region LPIPS` chưa phải artifact hoàn chỉnh trong submission hiện tại.
- Module 3 chưa có đánh giá định lượng cho identity preservation.

Chi tiết claim/doc safety của Phase 1A được ghi tại `docs/PHASE1A_CLAIM_SAFETY_CHANGELOG.md`.
