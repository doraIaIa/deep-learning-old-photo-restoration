# External Dependencies

## LaMa

Submission repo không vendor official LaMa vào GitHub. Repo research hiện đang gọi LaMa ngoài repo qua đường dẫn external và chạy bằng `conda run`.

- `repo_root`: thư mục source LaMa, ví dụ `F:/deeplearning/external_models/lama/lama`
- `checkpoint`: checkpoint Big-LaMa, ví dụ `F:/deeplearning/external_models/lama/weights/big-lama/models/best.ckpt`
- `conda_env_preferred`: environment GPU ưu tiên, hiện tại là `lama_gpu`
- `conda_env_fallback`: environment CPU fallback, hiện tại là `lama`
- `configs/external_paths.example.yaml` chỉ là template
- `configs/external_paths.yaml` là file local theo máy, đã được ignore trong Git

## CodeFormer

Submission repo cũng không vendor CodeFormer hoặc weight vào GitHub.

- `repo_root`: ví dụ `F:/deeplearning/external_models/CodeFormer`
- `checkpoint`: ví dụ `F:/deeplearning/external_models/CodeFormer/weights/CodeFormer/codeformer.pth`
- `conda_env`: ví dụ `codeformer`

## Vì sao không vendor vào GitHub

- Model repo ngoài submission giúp repo nộp gọn hơn.
- Checkpoint lớn không phù hợp để commit vào Git.
- Dependency ngoài repo giúp tách biệt code nộp bài và tài nguyên runtime cục bộ.

## Cách chỉnh config

Sao chép `configs/external_paths.example.yaml` thành file config cục bộ riêng và sửa các path theo máy đang chạy. Không hard-code các absolute path này sâu trong code mới.

## Smoke test LaMa

Phase hiện tại chỉ smoke official LaMa bằng golden mask, chưa đụng segmentation mới.

```bash
python scripts/smoke_lama_inpainting.py
```

Script sẽ:

- đọc `configs/external_paths.yaml`
- probe readiness của `lama_gpu` rồi `lama`
- chạy official LaMa bằng `demo3.png` và golden `final_mask.png`
- ghi output vào `examples/outputs/lama_smoke_demo3`
- lưu `lama_smoke_report.json` để so với golden `restored_before_face.png`

## Ghi chú import path

Repo submission hiện chưa dùng editable install. File `sitecustomize.py` đang hỗ trợ Python resolve package trong `src/` khi chạy script trực tiếp từ repo root. Đây là giải pháp tạm sạch cho phase hiện tại; về sau có thể thay bằng `pip install -e .`.
