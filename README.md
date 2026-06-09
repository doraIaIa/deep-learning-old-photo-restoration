# Old Photo Restoration Blueprint 2.1 Submission

Repo nộp đồ án Deep Learning về phục hồi ảnh cũ theo Blueprint 2.1. Đây là repo submission sạch được tách từ repo research cũ để giữ code gọn, dễ đọc, và dễ đánh giá.

## Phát biểu bài toán

Mục tiêu là khôi phục ảnh cũ bằng pipeline gồm phân đoạn vùng hư hại, tinh chỉnh mask, inpainting, và tùy chọn phục hồi khuôn mặt. Repo này ưu tiên tái lập behavior của repo gốc bằng golden artifacts thay vì mở rộng thuật toán.

## Các module Blueprint 2.1

- Module 1: phát hiện và phân đoạn vùng cần sửa.
- Module 2: tinh chỉnh mask để tăng độ phủ vùng hỏng.
- Module 3: inpainting bằng backend ngoài repo.
- Module 4: phục hồi khuôn mặt khi được bật.
- Module 5: đánh giá và so sánh bằng artifact/mask/metric cơ bản.

## Trạng thái submission hiện tại

- Phase hiện tại mới dựng skeleton sạch cho submission.
- Chưa full implementation pipeline mới.
- Chưa chạy inference từ repo mới.
- Golden reference đã được sinh từ repo research cũ và copy vào `examples/golden/demo3_r013_repair_wide/`.

## Golden demo reference

- Input tham chiếu: `examples/inputs/demo3.png`
- Golden output tham chiếu:
  - `examples/golden/demo3_r013_repair_wide/restored_before_face.png`
  - `examples/golden/demo3_r013_repair_wide/final_mask.png`
  - `examples/golden/demo3_r013_repair_wide/metadata.json`

## Đặt checkpoint

Checkpoint không được commit vào Git. Hãy đặt checkpoint segmentation tại:

`checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth`

Sau đó kiểm tra lại SHA256 với `configs/checkpoints.yaml`.

## External dependencies

LaMa và CodeFormer là dependency ngoài repo, không vendored trong submission này. Xem:

- `configs/external_paths.example.yaml`
- `configs/external_paths.yaml` là file local theo máy chạy, không commit
- `docs/external_dependencies.md`

## Lệnh dự kiến

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir examples/outputs/demo3
python scripts/run_gradio_demo.py
python scripts/run_ablation.py
python scripts/evaluate_segmentation.py
python scripts/smoke_lama_inpainting.py
```

## Cảnh báo

Repo này là repo submission sạch được migrate từ repo research. Phase hiện tại chỉ chuẩn bị cấu trúc, config, golden artifacts, và khung mã nguồn tối thiểu để tiếp tục triển khai từng bước.

## Trạng thái phase 1B

- Đã có config loader đọc `inference`, `checkpoints`, và `external_paths`.
- Đã có official LaMa wrapper đọc path từ config, không hard-code absolute path trong source.
- Phase này chỉ smoke LaMa bằng `examples/inputs/demo3.png` và golden `final_mask.png`.
- Chưa triển khai segmentation mới.
- Chưa full pipeline end-to-end trong repo submission.

## Trạng thái phase 1C

- Đã có `RestorationPipeline` cho đường chạy mask-bypass.
- `scripts/run_pipeline.py` chạy end-to-end khi truyền sẵn `--mask`.
- Nếu không truyền `--mask`, pipeline fail rõ vì segmentation chưa được triển khai trong Phase 1C.
- `sitecustomize.py` đang giúp Python resolve `src/` khi chạy script trực tiếp từ repo root.
- Khi cần chuẩn hóa hơn có thể thay bằng `pip install -e .`.
