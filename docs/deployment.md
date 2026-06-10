# Deployment

> **Tóm tắt tiếng Việt**
> 
> - Tài liệu này cung cấp hướng dẫn khởi chạy nhanh (deployment skeleton) thông qua Docker hoặc Local cho mục đích demo.
> - Image không đóng gói sẵn các model weights (LaMa, CodeFormer, Segmentation checkpoint) để tối ưu kích thước repo.
> - Vui lòng cấu hình artifact theo hướng dẫn nếu muốn chạy thử.

## Scope

This is a Docker/local deployment skeleton for demo and readiness purposes. The image does not bundle:

- the segmentation checkpoint;
- the official LaMa source or weights;
- the CodeFormer source or weights;
- datasets, research logs, or large runtime outputs.

## Safe Deployment Policy

- The project repository does not commit checkpoint binaries to Git by default.
- External runtimes and checkpoints should be mounted or referenced from outside the repository.
- This document describes policy and expected setup only; it does not modify the user's runtime configuration automatically.

## Checkpoint Reference

- The strongest checkpoint reference is `R013_REPRO`.
- The external artifact is configured through a local artifact root, for example:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- SHA256:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`

## Docker and Local Run

```bash
python scripts/check_readiness.py
python scripts/run_gradio_demo.py
```

```bash
docker compose up --build
```

## Caveats

- Docker is a deployment skeleton, not a self-contained production image.
- GPU/CUDA runtime setup is machine-specific.
- LaMa is described here as an official/pretrained external dependency, not as a fine-tuned model shipped by this repository.
