<div align="center">

# Phục Hồi Ảnh Cũ với Deep Learning

### Pipeline mô-đun: phát hiện vết nứt, xóa vết nứt, phục hồi màu và phục hồi khuôn mặt

**Crack Segmentation · Hybrid Mask · LaMa Inpainting · Color Restoration U-Net · CCM · CodeFormer**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-f97316?style=for-the-badge)](http://127.0.0.1:7860)
[![Docker](https://img.shields.io/badge/Docker-Template-2496ed?style=for-the-badge&logo=docker&logoColor=white)](docs/deployment.md)

<br>

<img src="docs/assets/demo3/contact_sheet.png" alt="Minh họa pipeline phục hồi ảnh cũ" width="930"/>

</div>

---

## 1. Tổng quan

Dự án giải quyết bài toán phục hồi ảnh cũ bị nứt, trầy xước, phai màu và suy giảm chất lượng. Thay vì sử dụng một mô hình end-to-end duy nhất, hệ thống chia bài toán thành các mô-đun độc lập để dễ kiểm tra, thay thế và đánh giá.

Đầu vào chính là một ảnh cũ RGB. Đầu ra chính là ảnh đã được:

1. Phát hiện vùng nứt hoặc hư hỏng.
2. Tạo mask sửa chữa phù hợp cho inpainting.
3. Xóa vết nứt bằng LaMa.
4. Phục hồi màu bằng OpenCV, Color Restoration U-Net, Inference Control và CCM.
5. Tùy chọn phục hồi khuôn mặt bằng CodeFormer.

Đây là một pipeline inference hoàn chỉnh, đồng thời chứa mã huấn luyện và đánh giá cho mô-đun phân vùng vết nứt.

### Nguồn gốc các mô hình

| Thành phần | Vai trò | Nguồn mô hình |
|---|---|---|
| Crack Segmentor R013/R014 | Phát hiện vùng nứt, trầy xước | Mô hình do dự án huấn luyện |
| LaMa | Xóa vùng nứt theo mask | Mô hình pretrained bên ngoài, không fine-tune trong repo |
| Color Restoration U-Net | Phục hồi màu và sắc độ | Mô hình do dự án huấn luyện, checkpoint lưu local |
| CodeFormer | Phục hồi khuôn mặt | Mô hình pretrained bên ngoài, tùy chọn |

Checkpoint và source tree của các external model không được commit vào Git.

---

## 2. Kiến trúc pipeline

```text
Ảnh cũ RGB
    |
    v
Crack Segmentor
    |
    +--> DL mask
    |
    v
Classical CV Crack Detector
    |
    +--> CV mask
    |
    v
Hybrid Mask + repair_wide_v1
    |
    +--> final_mask.png
    |
    v
LaMa Inpainting
    |
    +--> inpainting/lama_restored.png
    |
    v
PostInpaintingProcessor
    |
    +--> Color Restoration
    |      |
    |      +--> OpenCV conservative cleanup
    |      +--> Color Restoration U-Net
    |      +--> Inference Control
    |      +--> CCM color correction
    |      +--> Safety post-processing
    |
    +--> Face Restoration
    |      |
    |      +--> CodeFormer, nếu được bật
    |
    v
final/restored.png
```

### Trách nhiệm của `PostInpaintingProcessor`

`PostInpaintingProcessor` không phải là mô hình Deep Learning. Đây là lớp điều phối các bước sau LaMa:

- Gọi mô-đun phục hồi màu.
- Truyền kết quả phục hồi màu sang mô-đun phục hồi khuôn mặt.
- Lưu ảnh trung gian theo từng mô-đun.
- Tạo ảnh cuối cùng.
- Ghi metadata và log cho từng lần chạy.

CodeFormer thuộc mô-đun `face_restoration`, không thuộc mô-đun `color_restoration`.

---

## 3. Các mô-đun chính

### 3.1. Phân vùng vết nứt

Mô-đun segmentation nhận ảnh RGB và dự đoán probability mask của vùng cần sửa.

- `R013`: Custom Attention U-Net, là baseline vận hành mặc định.
- `R014`: Biến thể ResNet-34 thử nghiệm, có metric segmentation cao hơn nhưng không mặc định vì có thể làm giảm chất lượng pipeline end-to-end.
- Mask Deep Learning được kết hợp với nhánh Classical CV để giảm bỏ sót các vết nứt mảnh.
- Chính sách `repair_wide_v1` kết nối và làm rộng mask trước khi đưa vào LaMa.

Các ảnh trung gian quan trọng:

```text
dl_mask.png
cv_mask.png
union_before_refine.png
final_mask.png
```

### 3.2. Xóa vết nứt bằng LaMa

LaMa được gọi thông qua wrapper và chạy như một external runtime.

- Repo không chứa source tree chính thức của LaMa.
- Repo không fine-tune LaMa.
- Đường dẫn repo và checkpoint LaMa được cấu hình trong `configs/external_paths.yaml`.

Output chính:

```text
inpainting/lama_restored.png
```

### 3.3. Phục hồi màu

Mô-đun phục hồi màu nhận ảnh sau LaMa và thực hiện:

```text
Ảnh sau LaMa
-> OpenCV conservative cleanup
-> Color Restoration U-Net
-> Inference Control
-> CCM color correction
-> Safety post-processing
-> color_restored.png
```

#### Color Restoration U-Net

Mô hình hỗ trợ các chế độ:

- `lab_residual`: dự đoán residual cho các kênh Lab, là chế độ checkpoint hiện tại.
- `lab_ab`: dự đoán trực tiếp hai kênh màu `a`, `b`.
- `rgb_residual`: dự đoán residual trong không gian RGB.

Checkpoint hiện được cấu hình tại:

```text
checkpoints/color_restoration/color-mixed-lab-residual-v2-r001/best.pth
```

Checkpoint phục hồi màu không được commit vào Git. Cấu hình runtime nằm tại:

```text
configs/color_restoration.yaml
```

#### Inference Control

Inference Control giới hạn mức thay đổi độ sáng và màu để tránh mô hình tạo màu quá mạnh hoặc làm thay đổi ảnh ngoài ý muốn.

#### CCM

Color Correction Matrix được áp dụng ngay sau Inference Control để hiệu chỉnh màu toàn cục. CCM vẫn thuộc mô-đun phục hồi màu.

Các output trung gian:

```text
color_restoration/
├── quality_restored.png
├── model_restored.png
├── inference_controlled.png
├── ccm_corrected.png
├── color_restored.png
└── color_restoration_metadata.json
```

### 3.4. Phục hồi khuôn mặt

CodeFormer là bước tùy chọn chạy sau phục hồi màu.

```text
color_restored.png
-> CodeFormer
-> face_restoration/codeformer_output.png
```

Nếu CodeFormer bị tắt hoặc thiếu external runtime, mô-đun thực hiện pass-through và ghi lý do vào metadata. Dự án không cam kết bảo toàn hoàn toàn danh tính khuôn mặt.

---

## 4. Cài đặt môi trường

### 4.1. Sao chép repository

```bash
git clone https://github.com/doraIaIa/deep-learning-old-photo-restoration.git
cd deep-learning-old-photo-restoration
```

### 4.2. Tạo môi trường Python

Có thể sử dụng Conda:

```bash
conda create -n AIC23 python=3.10
conda activate AIC23
pip install -r requirements.txt
```

Hoặc sử dụng `venv`:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux hoặc macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Cấu hình mô hình bên ngoài và checkpoint

### 5.1. Tạo cấu hình local

Windows:

```powershell
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

Linux hoặc macOS:

```bash
cp configs/external_paths.example.yaml configs/external_paths.yaml
```

Chỉnh `configs/external_paths.yaml` để trỏ tới LaMa và CodeFormer trên máy local:

```yaml
lama:
  repo_root: <đường-dẫn-tới-repo-lama>
  checkpoint: <đường-dẫn-tới-best.ckpt>
  conda_env_preferred: lama_gpu
  conda_env_fallback: lama

codeformer:
  repo_root: <đường-dẫn-tới-repo-codeformer>
  checkpoint: <đường-dẫn-tới-codeformer.pth>
  conda_env: codeformer
```

`configs/external_paths.yaml` chứa đường dẫn riêng của từng máy và không nên commit.

### 5.2. Checkpoint Segmentor R013

Đường dẫn mặc định:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

SHA256 kỳ vọng:

```text
a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725
```

### 5.3. Checkpoint Segmentor R014

R014 là biến thể thử nghiệm:

```text
checkpoints/segmenter/seg-unet-resnet34-r014-s42/best_val_iou.pth
```

Cũng có thể cấu hình bằng biến môi trường `R014_SEGMENTER_CHECKPOINT`.

### 5.4. Checkpoint phục hồi màu

Đường dẫn được khai báo trong `configs/color_restoration.yaml`:

```text
checkpoints/color_restoration/color-mixed-lab-residual-v2-r001/best.pth
```

SHA256 kỳ vọng:

```text
a32ff2975967d5b2cd81634c5f4ef026bb184892b10de38a70e37961f9927c91
```

---

## 6. Kiểm tra trước khi chạy

Kiểm tra pipeline cốt lõi:

```bash
python scripts/check_readiness.py --strict
```

Kiểm tra cả mô-đun phục hồi màu:

```bash
python scripts/check_readiness.py --post-inpainting --strict
```

Kiểm tra artifact và checkpoint:

```bash
python scripts/verify_artifacts.py check-all --repo-root .
```

Chạy test:

```bash
python -m pytest -q
```

---

## 7. Chạy inference

### 7.1. Nơi đặt ảnh mới

Ảnh mới dùng để inference có thể đặt trong:

```text
inference_inputs/
```

Thư mục `data/raw/` dành cho dữ liệu huấn luyện hoặc dữ liệu nguồn, không phải nơi mặc định để upload ảnh inference.

### 7.2. Chạy pipeline cơ bản

Pipeline cơ bản chạy segmentation và LaMa, không chạy phục hồi màu:

```bash
python scripts/run_pipeline.py \
  --image inference_inputs/old_photo_001.jpg \
  --output-dir outputs/runs/basic_run
```

### 7.3. Chạy full pipeline với phục hồi màu

```bash
python scripts/run_pipeline.py \
  --image inference_inputs/old_photo_001.jpg \
  --output-dir outputs/runs/full_run \
  --post-inpainting
```

### 7.4. Chạy full pipeline và bật CodeFormer

```bash
python scripts/run_pipeline.py \
  --image inference_inputs/old_photo_001.jpg \
  --output-dir outputs/runs/full_run_codeformer \
  --post-inpainting \
  --face-mode auto
```

`--face-mode auto` yêu cầu `--post-inpainting`.

### 7.5. Chạy nhiều ảnh

```bash
python scripts/run_pipeline.py \
  --image inference_inputs/old_photo_001.jpg inference_inputs/old_photo_002.jpg \
  --output-dir outputs/runs/batch_run \
  --post-inpainting
```

Mỗi ảnh được tạo một item riêng dựa trên tên file.

### 7.6. Chạy với mask có sẵn

```bash
python scripts/run_pipeline.py \
  --image inference_inputs/old_photo_001.jpg \
  --mask path/to/final_mask.png \
  --output-dir outputs/runs/mask_bypass_run \
  --post-inpainting
```

### 7.7. Chạy phục hồi màu độc lập

Chạy model phục hồi màu trên ảnh đã qua LaMa:

```bash
python scripts/run_color_restoration.py \
  --input path/to/lama_restored.png \
  --output-dir outputs/color_restoration_standalone \
  --method model
```

Chạy smoke test chỉ dùng OpenCV, không yêu cầu checkpoint màu:

```bash
python scripts/run_color_restoration.py \
  --input path/to/lama_restored.png \
  --output-dir outputs/color_restoration_opencv \
  --method opencv_conservative
```

### 7.8. Chạy Gradio

```bash
python scripts/run_gradio_demo.py
```

Mở:

```text
http://127.0.0.1:7860
```

---

## 8. Cấu trúc output

`run_pipeline.py` xem `--output-dir` là thư mục của một batch. Với ảnh `old_photo_001.jpg`, output có dạng:

```text
outputs/runs/full_run/
├── batch_manifest.json
└── items/
    └── old_photo_001/
        ├── input/
        │   └── original.jpg
        ├── artifacts/
        │   ├── dl_mask.png
        │   ├── cv_mask.png
        │   ├── union_before_refine.png
        │   ├── final_mask.png
        │   ├── inpainting/
        │   │   └── lama_restored.png
        │   ├── color_restoration/
        │   │   ├── quality_restored.png
        │   │   ├── model_restored.png
        │   │   ├── inference_controlled.png
        │   │   ├── ccm_corrected.png
        │   │   ├── color_restored.png
        │   │   └── color_restoration_metadata.json
        │   ├── face_restoration/
        │   │   ├── codeformer_output.png
        │   │   └── face_restoration_metadata.json
        │   ├── final/
        │   │   └── restored.png
        │   ├── logs/
        │   │   └── pipeline.log
        │   └── metadata.json
        ├── final.png
        └── manifest.json
```

Các alias tương thích có thể vẫn xuất hiện ở thư mục `artifacts/`, ví dụ `lama_restored.png`, `restored_before_face.png`, `face_restored.png` và `restored_color.png`.

Nếu post-inpainting bị tắt, `final.png` là kết quả sau LaMa. Nếu CodeFormer bị tắt, `final.png` là kết quả sau phục hồi màu.

---

## 9. Demo3 và kết quả segmentation

Repo cung cấp `demo3` như một golden regression case để kiểm tra pipeline. Đây là minh chứng định tính có kiểm soát, không phải benchmark đại diện cho toàn bộ miền ảnh cũ thực tế.

### Kết quả R013

R013 được fine-tune trên 118 cặp ảnh-mask hợp lệ.

| Metric | R011 baseline | R013 selected checkpoint |
|---|---:|---:|
| IoU | 0.2527 | **0.3457** |
| F1 | 0.4025 | **0.5097** |
| Precision | 0.4112 | **0.5887** |
| Recall | 0.4083 | 0.4670 |
| Val IoU | - | 0.3812 |
| Val F1 | - | 0.5503 |

Các metric trên đánh giá mô-đun segmentation, không phải điểm benchmark đầy đủ của pipeline phục hồi ảnh.

### Demo gallery

<p align="center">
  <img src="docs/assets/demo3/input.png" width="280" alt="Ảnh đầu vào"/>
  <img src="docs/assets/demo3/final_mask.png" width="280" alt="Mask sửa chữa cuối"/>
  <img src="docs/assets/demo3/restored_before_face.png" width="280" alt="Ảnh sau phục hồi"/>
</p>

---

## 10. Cấu trúc repository

```text
deep-learning-old-photo-restoration/
├── app/
│   └── gradio_demo.py
├── artifacts/
│   └── manifests/
├── checkpoints/
│   ├── segmenter/
│   └── color_restoration/
├── configs/
│   ├── checkpoints.yaml
│   ├── color_restoration.yaml
│   ├── external_paths.example.yaml
│   └── inference.yaml
├── data/
│   ├── manifests/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── docs/
├── examples/
├── inference_inputs/
├── scripts/
├── src/old_photo_restoration/
│   ├── color_restoration/
│   ├── evaluation/
│   ├── face_restoration/
│   ├── inpainting/
│   ├── segmentation/
│   ├── utils/
│   ├── pipeline.py
│   └── postprocessing.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Điểm chạy chính

| Script | Chức năng |
|---|---|
| `scripts/run_pipeline.py` | Chạy pipeline trên một hoặc nhiều ảnh |
| `scripts/run_color_restoration.py` | Chạy phục hồi màu độc lập |
| `scripts/run_gradio_demo.py` | Khởi động giao diện Gradio |
| `scripts/check_readiness.py` | Kiểm tra dependency, config và checkpoint |
| `scripts/verify_artifacts.py` | Kiểm tra artifact và checkpoint policy |
| `scripts/evaluate_segmentation.py` | Đánh giá mô-đun segmentation |
| `scripts/eval_pipeline_paired.py` | Đánh giá pipeline trên dữ liệu paired |
| `scripts/train_segmentation.py` | Entry point huấn luyện segmentation |

---

## 11. Quy ước và giới hạn

- **Segmentor**: repo có mã huấn luyện và đánh giá; checkpoint binary không được commit.
- **LaMa**: chỉ dùng pretrained external runtime; repo không claim fine-tune LaMa.
- **Color Restoration U-Net**: repo chứa kiến trúc và runtime inference; checkpoint được cung cấp local.
- **CodeFormer**: external model tùy chọn; không cam kết bảo toàn danh tính.
- **Checkpoint và output**: được ignore khỏi Git để giữ repository nhẹ.
- **Demo3**: chỉ là golden regression case, không phải benchmark cấp tập dữ liệu.
- **LPIPS/FID**: không được claim là kết quả đánh giá hoàn chỉnh nếu chưa có artifact kiểm chứng.

Xem thêm:

- [Kiến trúc hệ thống](docs/architecture.md)
- [Phụ thuộc bên ngoài](docs/external_dependencies.md)
- [Giới hạn dự án](docs/limitations.md)
- [Khả năng tái tạo](docs/reproducibility.md)
- [Phạm vi claim](docs/scope_and_claim_safety.md)

---

## 12. Docker

Docker trong repo đóng vai trò template môi trường cho phần mã chính. LaMa và CodeFormer vẫn là external runtime và cần được mount hoặc cấu hình riêng tùy máy.

Xem hướng dẫn tại [docs/deployment.md](docs/deployment.md).

---

## 13. Ghi nhận phụ thuộc bên ngoài

Dự án sử dụng:

- [LaMa](https://github.com/advimman/lama) làm inpainting backend.
- [CodeFormer](https://github.com/sczhou/CodeFormer) cho bước phục hồi khuôn mặt tùy chọn.

Khi phân phối source, pretrained weights hoặc artifact phát sinh, cần tuân thủ license và yêu cầu trích dẫn của từng dự án gốc.

---

## Giấy phép

Dự án học thuật cho môn Deep Learning. Kiểm tra license của các external dependency và pretrained weight trước khi tái phân phối hoặc sử dụng ngoài phạm vi học thuật.
