# Deployment

## Scope

Đây là Docker/local deployment skeleton cho mục đích demo hoặc readiness check. Image không tự chứa:

- checkpoint segmentation;
- official LaMa source hoặc weight;
- CodeFormer source hoặc weight;
- dataset, logs hoặc output research.

## Chính sách triển khai an toàn

- Repo submission không commit checkpoint vào Git theo mặc định.
- External runtime và checkpoint nên được mount hoặc tham chiếu từ bên ngoài repo.
- Tài liệu này chỉ nêu rõ policy và không tự thay đổi config runtime của người dùng.

## Checkpoint tham chiếu

- Checkpoint tham chiếu có bằng chứng tái lập mạnh nhất hiện tại là `R013_REPRO`.
- Artifact ngoài repo được cấu hình qua local artifact root, ví dụ:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- SHA256:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`

## Docker Và Local Run

```bash
python scripts/check_readiness.py
python scripts/run_gradio_demo.py
```

```bash
docker compose up --build
```

## Caveats

- Docker ở trạng thái skeleton, không phải image production self-contained.
- GPU/CUDA runtime cần cấu hình riêng theo máy.
- LaMa trong repo submission được mô tả là official/pretrained external dependency, không phải LaMa fine-tune.
