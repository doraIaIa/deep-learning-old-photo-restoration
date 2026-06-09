# Old Photo Restoration using Deep Learning

Pipeline phục hồi ảnh cũ bằng:
- r013 crack/defect segmentation
- CV crack mask builder
- hybrid union mask
- `repair_wide_v1` refinement
- official LaMa inpainting
- CodeFormer optional, chưa bật trong current pipeline nếu chưa implement

## Pipeline

`Input photo -> Segmentation -> CV mask -> Union -> Mask refinement -> LaMa -> Restored image`

## Current Status

- Phase 1C mask-bypass pass với `MAE = 0`
- Phase 2 auto-mask pass với `IoU = 0.9997728217` so với golden demo3
- Checkpoint và external weights không được commit vào Git

## Setup

1. Tạo Python environment.
2. Cài dependency:

```bash
pip install -r requirements.txt
```

3. Tạo config local:

```bash
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

4. Chỉnh path LaMa trong `configs/external_paths.yaml`.
5. Đặt checkpoint r013 vào:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

6. Verify SHA256 checkpoint:

```text
a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725
```

## Readiness Check

```bash
python scripts/check_readiness.py
```

## Run Local Gradio Demo

```bash
python scripts/check_readiness.py
python app/gradio_demo.py
```

Hoặc:

```bash
python scripts/run_gradio_demo.py
```

Sau đó upload ảnh và chạy auto-mask pipeline. Output sẽ nằm trong `examples/outputs/gradio_runs/`.
Thư mục `examples/outputs/` đang bị gitignore.

## Deployment / Docker Skeleton

- Repo đã có `Dockerfile` và `docker-compose.yml`
- Skeleton này không self-contained weights hoặc external model runtime
- Xem thêm `docs/deployment.md`
- Lệnh nhanh:

```bash
docker compose up --build
```

## Run Mask-Bypass Smoke

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --mask examples/golden/demo3_r013_repair_wide/final_mask.png --output-dir examples/outputs/pipeline_smoke_demo3 --face-mode off --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png
```

## Run Auto-Mask Pipeline

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir examples/outputs/seg_smoke_demo3 --face-mode off --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

## Expected Demo3 Metrics

- `final_mask_ratio` khoảng `0.09799`
- `final_mask IoU vs golden` khoảng `0.99977`
- `restored PSNR` khoảng `66.64 dB`

## Repository Policy

- `checkpoints/` bị ignore
- `configs/external_paths.yaml` bị ignore
- `examples/outputs/` bị ignore
- golden artifacts được giữ trong repo để tái lập
