# External Dependencies

## LaMa

Repo submission không vendor official LaMa vào cây source. Pipeline gọi LaMa qua wrapper/external runtime.

- Đây là official/pretrained integration.
- Không claim LaMa đã fine-tune trong repo submission này.
- `configs/external_paths.example.yaml` chỉ là template.
- `configs/external_paths.yaml` là file local theo máy và không nên commit.

## Checkpoint Policy

- Checkpoint segmentation không commit vào Git theo mặc định.
- Artifact tham chiếu ngoài repo cho `R013_REPRO` được cấu hình qua local artifact root, ví dụ:
  `<LOCAL_ARTIFACT_ROOT>/module1_retrain_sequence/R013_REPRO/best_iou.ckpt`
- SHA256:
  `5f3b340e38eba8290d2b8ca030bb51126308169f4e42087f46ddac0334e74203`

## CodeFormer

- CodeFormer là dependency optional.
- Trong repo submission hiện tại, Module 3 không nên được mô tả như một capability end-to-end đã hoàn chỉnh.
- Nếu có demo qualitative ở ngữ cảnh khác, chỉ nên mô tả ở mức qualitative hoặc prototype.
- Không claim identity preservation.

## Vì Sao Không Commit External Dependencies

- Giữ repo submission nhẹ và dễ review.
- Tránh hard-code path local trong source tree.
- Dùng `configs/external_paths.example.yaml` và các manifest dưới `artifacts/manifests/` để map local runtime paths.
- Tách source code khỏi weights, runtime và artifact research.
