<div align="center">

# Phục Hồi Ảnh Cũ với Deep Learning

### Modular Old Photo Restoration Pipeline

**Damage Segmentation · Hybrid Mask Refinement · LaMa Inpainting · Color Restoration · Optional Face Restoration**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-f97316?style=for-the-badge)](docs/demo_script.md)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Template-2496ed?style=for-the-badge&logo=docker&logoColor=white)](docs/deployment.md)

<br>

<img src="docs/assets/demo3/contact_sheet.png" alt="Minh họa pipeline phục hồi ảnh cũ" width="930"/>

<sub><b>Ảnh đầu vào</b> &nbsp;→&nbsp; <b>Repair mask dự đoán</b> &nbsp;→&nbsp; <b>Kết quả sau LaMa inpainting</b></sub>

</div>

---

## Tổng quan

Dự án xây dựng một pipeline phục hồi ảnh cũ theo kiến trúc mô-đun, tập trung vào các hư hại có cấu trúc như vết nứt, trầy xước và rách nhỏ. Thay vì giao toàn bộ bài toán cho một mô hình end-to-end duy nhất, hệ thống tách từng trách nhiệm thành các module độc lập để có thể quan sát, kiểm thử, thay thế và đánh giá riêng.

Pipeline cốt lõi phát hiện vùng hư hại, tạo repair mask và dùng pretrained LaMa để inpainting. Khi bật `post-inpainting`, hệ thống tiếp tục phục hồi chất lượng và màu sắc; CodeFormer có thể được bật như một bước phục hồi khuôn mặt tùy chọn.

### Điểm nổi bật

- **Pipeline mô-đun và dễ kiểm tra**: mỗi stage lưu artifact, metadata và log riêng.
- **Hybrid repair mask**: kết hợp Deep Learning segmentation với Classical Computer Vision.
- **Hai segmenter**: R013 ổn định mặc định và R014 ResNet-34 thử nghiệm.
- **Post-inpainting đầy đủ**: quality restoration, Color Restoration U-Net, inference control, CCM và safety post-processing.
- **Nhiều cách sử dụng**: CLI một ảnh, batch inference, mask-bypass, Gradio và Docker template.
- **Artifact-driven reproducibility**: checkpoint, dataset và experiment lineage được quản lý qua manifest.
- **Claim-safety rõ ràng**: phân biệt giữa khả năng đã triển khai, kết quả thực nghiệm và giới hạn chưa được chứng minh.

## Mục lục

- [Phạm vi và trạng thái module](#phạm-vi-và-trạng-thái-module)
- [Kiến trúc pipeline](#kiến-trúc-pipeline)
- [Demo trực quan](#demo-trực-quan)
- [Kết quả và đánh giá](#kết-quả-và-đánh-giá)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt và cấu hình](#cài-đặt-và-cấu-hình)
- [Chạy inference](#chạy-inference)
- [Cấu trúc output](#cấu-trúc-output)
- [Huấn luyện, đánh giá và kiểm thử](#huấn-luyện-đánh-giá-và-kiểm-thử)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Reproducibility và artifact policy](#reproducibility-và-artifact-policy)
- [Giới hạn và claim-safety](#giới-hạn-và-claim-safety)
- [Tài liệu liên quan](#tài-liệu-liên-quan)
- [Đóng góp](#đóng-góp)
- [Ghi nhận và giấy phép](#ghi-nhận-và-giấy-phép)

---

## Phạm vi và trạng thái module

| Module | Vai trò | Trạng thái | Output chính |
|---|---|---|---|
| Damage Segmentation | Phát hiện crack và vùng hư hại | Đã triển khai; R013 là mặc định | `dl_mask.png` |
| Classical CV Detection | Bổ sung các crack mảnh, tương phản | Đã triển khai | `cv_mask.png` |
| Hybrid Mask Refinement | Union mask và chính sách `repair_wide_v1` | Đã triển khai | `final_mask.png` |
| LaMa Inpainting | Tái tạo vùng bị che bởi repair mask | External pretrained backend | `inpainting/lama_restored.png` |
| Color Restoration | Phục hồi chất lượng và màu sau LaMa | Đã tích hợp; bật bằng `--post-inpainting` | `color_restoration/color_restored.png` |
| Face Restoration | Phục hồi khuôn mặt bằng CodeFormer | Tùy chọn; bật bằng `--face-mode auto` | `face_restoration/codeformer_output.png` |
| Batch & Artifact Management | Tổ chức input, output, manifest và metadata | Đã triển khai | `batch_manifest.json`, `metadata.json` |

### Nguồn gốc mô hình

| Thành phần | Nguồn | Chính sách |
|---|---|---|
| Segmenter R013/R014 | Mô hình được huấn luyện trong phạm vi dự án | Checkpoint được lưu local, không commit vào Git |
| LaMa | Official/pretrained external dependency | Repo không claim fine-tune LaMa |
| Color Restoration U-Net | Kiến trúc và inference runtime thuộc dự án | Checkpoint được lưu local, không commit vào Git |
| CodeFormer | Pretrained external dependency | Tùy chọn, không đảm bảo identity preservation |

---

## Kiến trúc pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│                         ẢNH CŨ RGB                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ DAMAGE SEGMENTATION                                          │
│                                                              │
│ • R013 Custom Attention U-Net: mặc định                      │
│ • R014 ResNet-34 Attention U-Net: thử nghiệm                 │
│                                                              │
│ Output: dl_mask.png                                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ CLASSICAL CV CRACK DETECTION                                 │
│                                                              │
│ • CLAHE                                                      │
│ • Blackhat / Canny / morphology                             │
│ • Component filtering                                       │
│                                                              │
│ Output: cv_mask.png                                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ HYBRID UNION MASK + REPAIR_WIDE_V1                           │
│                                                              │
│ Outputs:                                                     │
│ • union_before_refine.png                                    │
│ • final_mask.png                                             │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ OFFICIAL PRETRAINED LAMA WRAPPER                             │
│                                                              │
│ Output: inpainting/lama_restored.png                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
              CORE PIPELINE KẾT THÚC TẠI ĐÂY
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ POST-INPAINTING PROCESSOR                         [TÙY CHỌN] │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ COLOR RESTORATION                                        │ │
│ │                                                          │ │
│ │ • Quality restoration                                    │ │
│ │ • Color Restoration U-Net                                │ │
│ │ • Inference control                                      │ │
│ │ • CCM color correction                                   │ │
│ │ • Safety post-processing                                 │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ FACE RESTORATION / CODEFORMER                 [TÙY CHỌN] │ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUTS                                                      │
│                                                              │
│ • final/restored.png                                         │
│ • metadata.json                                              │
│ • pipeline.log                                               │
└──────────────────────────────────────────────────────────────┘
```

### Damage Segmentation

Segmenter nhận ảnh RGB và dự đoán probability mask cho vùng cần sửa.

- **R013 Custom Attention U-Net** là operational baseline mặc định.
- **R014 ResNet-34 Attention U-Net** có raw segmentation metric cao hơn trong thí nghiệm riêng, nhưng vẫn được giữ ở trạng thái experimental vì chất lượng mask tốt hơn không luôn đồng nghĩa với chất lượng restoration tốt hơn.
- `Attention Gate` giúp decoder tập trung vào vùng crack-like và hạn chế background noise.

### Hybrid Mask Refinement

Nhánh Classical CV bổ sung các crack mảnh mà segmenter có thể bỏ sót. Sau khi union `dl_mask` và `cv_mask`, chính sách `repair_wide_v1` áp dụng morphology để nối và làm rộng vùng hư hại trước khi gửi sang LaMa.

Repair mask được thiết kế cho mục tiêu inpainting thực dụng, không claim là pixel-perfect segmentation mask.

### LaMa Inpainting

LaMa được gọi qua external runtime wrapper và chỉ được sử dụng dưới dạng official/pretrained backend.

- Source tree và weights của LaMa không nằm trong repository.
- Repo không fine-tune LaMa.
- Runtime path được cấu hình qua `configs/external_paths.yaml`.

### Color Restoration

Khi bật `--post-inpainting`, ảnh sau LaMa đi qua các stage:

```text
lama_restored
→ quality_restoration
→ color_restoration_model
→ inference_control
→ ccm_color_correction
→ safety_postprocessing
→ color_restored
```

Module này tập trung vào phục hồi chất lượng và hiệu chỉnh màu một cách có kiểm soát.

Các intermediate được lưu mặc định:

```text
quality_restored.png
model_restored.png
inference_controlled.png
ccm_corrected.png
color_restored.png
color_restoration_metadata.json
```

### Optional Face Restoration

CodeFormer chạy sau color restoration khi người dùng đồng thời bật:

```text
--post-inpainting --face-mode auto
```

Nếu CodeFormer bị tắt hoặc external runtime chưa sẵn sàng, module thực hiện pass-through và ghi rõ trạng thái vào metadata. Dự án không đảm bảo CodeFormer bảo toàn hoàn toàn danh tính khuôn mặt.

---

## Demo trực quan

`demo3` là một controlled case study dùng cho demo, smoke test và regression inspection. Đây là bằng chứng định tính trên một trường hợp cố định, không phải benchmark đại diện cho toàn bộ miền ảnh cũ thực tế.

### Input, repair mask và kết quả sau LaMa

<p align="center">
  <img src="docs/assets/demo3/input.png" width="280" alt="Ảnh cũ đầu vào"/>
  &nbsp;
  <img src="docs/assets/demo3/final_mask.png" width="280" alt="Repair mask cuối"/>
  &nbsp;
  <img src="docs/assets/demo3/restored_before_face.png" width="280" alt="Kết quả sau LaMa"/>
</p>
<p align="center">
  <sub>Input image &nbsp;·&nbsp; Final repair mask &nbsp;·&nbsp; LaMa restored output</sub>
</p>

### Repair mask overlay

<p align="center">
  <img src="docs/assets/demo3/overlay_mask.png" width="650" alt="Repair mask overlay trên ảnh đầu vào"/>
</p>

Các artifact chẩn đoán chi tiết hơn nằm tại:

```text
outputs/project_evolution/demo3_case_study/
```

---

## Kết quả và đánh giá

### Segmentation R013

R013 bắt đầu từ 120 ảnh nhưng chỉ có **118 valid image-mask pairs**. Fixed split là `83 / 18 / 17`, và R013 được khởi tạo từ R011.

| Thiết lập | IoU | F1 |
|---|---:|---:|
| R011_REPRO fair test @ `0.55` | 0.246848 | 0.394876 |
| R013_REPRO fair test @ `0.50` | **0.337970** | **0.501339** |

| Validation R013 | IoU |
|---|---:|
| Historical validation | 0.381231 |
| Reproduced validation | 0.380532 |

Các số liệu trên đánh giá Module 1 segmentation, không phải full end-to-end restoration benchmark.

### R014 experimental segmenter

Trong thí nghiệm held-out riêng, R014 ResNet-34 đạt `IoU = 0.4506` và `F1 = 0.6213`. R014 được tích hợp vào CLI và Gradio như một lựa chọn thử nghiệm với threshold `0.30` và dilation radius `1` tương ứng kernel `3x3`.

R013 vẫn là lựa chọn mặc định ổn định. Xem chi tiết tại [docs/experiments/r014_resnet34_summary.md](docs/experiments/r014_resnet34_summary.md).

### Paired restoration evaluation

Đánh giá hiện có trên synthetic paired data cho thấy automatic hybrid mask vẫn là bottleneck chính:

| Metric | Degraded baseline | Auto/hybrid restored | Delta | Số mẫu cải thiện |
|---|---:|---:|---:|---:|
| Full-image PSNR | 19.127 | 17.355 | -1.772 | 0/30 |
| Full-image MAE | 24.984 | 27.989 | +3.005 | 0/30 |
| Masked-region MAE | 35.666 | 34.151 | -1.515 | 15/30 |

Kết quả cho thấy pipeline có tín hiệu sửa chữa cục bộ tại vùng hư hại, nhưng chưa cải thiện ổn định chất lượng toàn ảnh. Oracle-mask ablation củng cố nhận định rằng chất lượng automatic mask là điểm nghẽn quan trọng.

Xem diễn giải đầy đủ tại [docs/restoration_evaluation.md](docs/restoration_evaluation.md).

### Phạm vi đánh giá chưa được claim

- Chưa có benchmark đầy đủ trên tập ảnh cũ thực tế.
- Chưa claim LPIPS, FID hoặc masked-region LPIPS.
- Chưa có dataset-level benchmark riêng cho color restoration.
- `demo3` chỉ là golden regression case.

---

## Yêu cầu hệ thống

### Bắt buộc cho core pipeline

- Python `3.10`
- Git
- Các package trong `requirements.txt`
- Segmenter checkpoint R013 hoặc R014
- Official LaMa source tree và pretrained weights
- `configs/external_paths.yaml` phù hợp với máy local

### Tùy chọn

- CUDA-capable GPU để tăng tốc inference
- Color restoration checkpoint khi dùng `method: model`
- CodeFormer source tree và weights khi bật face restoration
- Docker và Docker Compose nếu dùng deployment template

> LaMa và CodeFormer có runtime/environment riêng. Việc chỉ cài `requirements.txt` của repository không thay thế bước thiết lập các external dependency này.

---

## Cài đặt và cấu hình

### 1. Clone repository

```bash
git clone https://github.com/doraIaIa/deep-learning-old-photo-restoration.git
cd deep-learning-old-photo-restoration
```

### 2. Tạo môi trường Python

Với `venv`:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux hoặc macOS:

```bash
source .venv/bin/activate
```

Cài dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Cấu hình external dependencies

Windows:

```powershell
Copy-Item configs\external_paths.example.yaml configs\external_paths.yaml
```

Linux hoặc macOS:

```bash
cp configs/external_paths.example.yaml configs/external_paths.yaml
```

Chỉnh `configs/external_paths.yaml` để trỏ tới LaMa và optional CodeFormer trên máy local:

```yaml
lama:
  repo_root: <path-to-lama-repository>
  checkpoint: <path-to-big-lama-best.ckpt>
  conda_env_preferred: lama_gpu
  conda_env_fallback: lama

codeformer:
  repo_root: <path-to-codeformer-repository>
  checkpoint: <path-to-codeformer.pth>
  conda_env: codeformer
```

`configs/external_paths.yaml` chứa đường dẫn riêng của từng máy và không nên được commit.

### 4. Chuẩn bị checkpoint

| Checkpoint | Vai trò | Đường dẫn mặc định | SHA256 |
|---|---|---|---|
| R013 | Segmenter mặc định | `checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth` | `a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725` |
| R014 | Segmenter thử nghiệm | `checkpoints/segmenter/seg-unet-resnet34-r014-s42/best_val_iou.pth` | `e1d84ced2e3aac6fd89bbe48bd6149cc445cc7308b03887ac3f66de2352924c2` |
| Color Restoration U-Net | Post-inpainting color restoration | `checkpoints/color_restoration/color-mixed-lab-residual-v2-r001/best.pth` | `a32ff2975967d5b2cd81634c5f4ef026bb184892b10de38a70e37961f9927c91` |

Checkpoint binaries, dataset và external model weights không được commit vào Git.

### 5. Kiểm tra readiness

Kiểm tra core pipeline:

```bash
python scripts/check_readiness.py --strict
```

Kiểm tra thêm color restoration:

```bash
python scripts/check_readiness.py --post-inpainting --strict
```

Kiểm tra manifest và artifact policy:

```bash
python scripts/verify_artifacts.py check-all --repo-root .
```

---

## Chạy inference

`run_pipeline.py` hỗ trợ một hoặc nhiều ảnh. Thư mục `--output-dir` được xem là một batch output; mỗi ảnh sẽ có item directory riêng.

### Core pipeline

Chạy segmentation, hybrid mask và LaMa:

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir outputs/demo3_core
```

### Full pipeline với color restoration

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir outputs/demo3_full --post-inpainting
```

### Full pipeline với optional CodeFormer

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir outputs/demo3_full_face --post-inpainting --face-mode auto
```

`--face-mode auto` yêu cầu bật `--post-inpainting`.

### Chạy R014 experimental segmenter

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir outputs/demo3_r014 --segmenter-arch r014_resnet34
```

R014 tự dùng threshold `0.30` và dilation radius `1` nếu người dùng không override.

### Chạy với repair mask có sẵn

Mask-bypass mode bỏ qua bước auto segmentation:

```bash
python scripts/run_pipeline.py --image examples/inputs/demo3.png --mask examples/golden/demo3_r013_repair_wide/final_mask.png --output-dir outputs/demo3_mask_bypass
```

### Chạy batch nhiều ảnh

```bash
python scripts/run_pipeline.py --image inference_inputs/photo_01.jpg inference_inputs/photo_02.jpg --output-dir outputs/batch_run --post-inpainting
```

Ảnh inference mới nên được đặt trong `inference_inputs/`. Thư mục `data/raw/` dành cho dữ liệu nguồn hoặc dữ liệu huấn luyện.

### Chạy color restoration độc lập

Chạy model:

```bash
python scripts/run_color_restoration.py --input path/to/lama_restored.png --output-dir outputs/color_restoration_run --method model
```

Chạy conservative OpenCV mode không yêu cầu color checkpoint:

```bash
python scripts/run_color_restoration.py --input path/to/lama_restored.png --output-dir outputs/color_restoration_opencv --method opencv_conservative
```

### Giao diện Gradio

```bash
python scripts/run_gradio_demo.py
```

Mở `http://127.0.0.1:7860`. Gradio hỗ trợ:

- Auto-mask pipeline
- R013 hoặc R014 segmenter
- Bật/tắt post-inpainting color restoration
- Bật/tắt optional face restoration
- Xem final mask, output, metadata và trạng thái runtime

### Docker

```bash
docker compose up --build
```

Docker image là deployment template, không phải self-contained production image. Người dùng vẫn cần mount hoặc cấu hình checkpoint, LaMa và CodeFormer phù hợp. Xem [docs/deployment.md](docs/deployment.md).

---

## Cấu trúc output

Ví dụ với ảnh `photo_01.jpg` và `--output-dir outputs/batch_run`:

```text
outputs/batch_run/
├── batch_manifest.json
└── items/
    └── photo_01/
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

Các thư mục color restoration, face restoration và final chỉ xuất hiện đầy đủ khi post-inpainting được bật. Một số compatibility alias vẫn được tạo trong `artifacts/` để hỗ trợ CLI, Gradio và golden-reference tooling.

Nên dùng các canonical artifact sau khi xây dựng gallery hoặc đánh giá:

| Stage | Canonical artifact |
|---|---|
| Repair mask | `artifacts/final_mask.png` |
| LaMa output | `artifacts/inpainting/lama_restored.png` |
| Color restoration output | `artifacts/color_restoration/color_restored.png` |
| CodeFormer output | `artifacts/face_restoration/codeformer_output.png` |
| Final output | `final.png` hoặc `artifacts/final/restored.png` |

---

## Huấn luyện, đánh giá và kiểm thử

### Kiểm thử

Repository có unit test và integration test cho color restoration, face restoration adapter, post-inpainting pipeline và batch output.

Nếu môi trường chưa có `pytest`:

```bash
python -m pip install pytest
```

Chạy toàn bộ test:

```bash
python -m pytest -q
```

### Segmentation evaluation

Xem các tùy chọn đánh giá mask:

```bash
python scripts/evaluate_segmentation.py masks --help
```

### Training và reproduction status

```bash
python scripts/train_segmentation.py status --repo-root .
```

R013-specific fine-tuning/reproduction entry point:

```bash
python scripts/train_r013_finetune.py status --repo-root .
```

Dataset và checkpoint phục vụ training reproduction là external artifacts; repository chỉ đóng gói source code, manifest và metadata cần thiết.

### Demo asset và artifact verification

```bash
python scripts/build_demo_assets.py
python scripts/verify_artifacts.py check-all --repo-root .
```

Xem danh sách script và phạm vi sử dụng tại [scripts/README.md](scripts/README.md).

---

## Cấu trúc repository

```text
deep-learning-old-photo-restoration/
├── app/                              # Gradio application
├── artifacts/manifests/              # Dataset, checkpoint và run manifests
├── checkpoints/                      # Local ignored model checkpoints
├── configs/
│   ├── checkpoints.yaml
│   ├── color_restoration.yaml
│   ├── external_paths.example.yaml
│   └── inference.yaml
├── data/                             # Dataset skeleton, manifests và splits
├── docs/                             # Architecture, evaluation và policy docs
├── examples/
│   ├── golden/                       # Frozen regression references
│   ├── inputs/
│   └── outputs/                      # Local generated examples
├── inference_inputs/                 # Local input images for inference
├── outputs/                          # Local runs và curated project evidence
├── scripts/                          # Public CLI entry points
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
│   ├── integration/
│   └── unit/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Entry point chính

| Script | Chức năng |
|---|---|
| `scripts/run_pipeline.py` | Chạy core hoặc full pipeline trên một hay nhiều ảnh |
| `scripts/run_color_restoration.py` | Chạy color restoration độc lập |
| `scripts/run_gradio_demo.py` | Khởi động giao diện Gradio |
| `scripts/check_readiness.py` | Kiểm tra config, dependency và checkpoint |
| `scripts/verify_artifacts.py` | Kiểm tra artifact manifest và checkpoint policy |
| `scripts/evaluate_segmentation.py` | Đánh giá predicted mask với ground truth |
| `scripts/train_segmentation.py` | Xem training reproduction metadata |
| `scripts/train_r013_finetune.py` | Entry point cho R013 fine-tuning/reproduction |

---

## Reproducibility và artifact policy

Repository ưu tiên khả năng tái kiểm tra tối thiểu mà không biến Git thành nơi lưu dataset, model weight hoặc runtime output dung lượng lớn.

### Golden regression case

```text
examples/golden/demo3_r013_repair_wide/
├── final_mask.png
├── metadata.json
└── restored_before_face.png
```

### Chính sách artifact

- Checkpoint binaries và datasets được lưu local hoặc trong external artifact store.
- `configs/external_paths.yaml` là cấu hình riêng của máy và không được commit.
- `artifacts/manifests/` mô tả provenance, trạng thái và chính sách của artifact.
- `examples/golden/` chứa frozen expected references.
- Generated output trong `examples/outputs/` và `outputs/` bị ignore theo mặc định.
- Chỉ curated project-evolution evidence được theo dõi có chủ đích.

### Tái tạo demo tối thiểu

```bash
python scripts/check_readiness.py --strict
python scripts/run_pipeline.py --image examples/inputs/demo3.png --output-dir outputs/demo3_replay --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

Xem chi tiết tại [docs/reproducibility.md](docs/reproducibility.md) và [docs/artifacts.md](docs/artifacts.md).

---

## Giới hạn và claim-safety

Để tránh diễn giải quá mức kết quả, repository áp dụng các ranh giới sau:

- **Automatic mask vẫn là bottleneck**: segmentation metric tốt hơn không đảm bảo restoration output tốt hơn.
- **LaMa là pretrained external backend**: repo không claim fine-tune LaMa.
- **Color restoration chưa có dataset-level benchmark được công bố**: module đã được tích hợp và kiểm thử về execution/artifact contract.
- **CodeFormer là tùy chọn**: không đảm bảo identity preservation.
- **Demo3 là controlled case study**: không đại diện cho toàn bộ miền ảnh cũ thực tế.
- **Full real-photo benchmark chưa hoàn chỉnh**.
- **LPIPS, FID và masked-region LPIPS chưa được claim**.
- **Docker chỉ là deployment template**: external models và weights không được đóng gói sẵn.

Xem đầy đủ tại:

- [docs/limitations.md](docs/limitations.md)
- [docs/scope_and_claim_safety.md](docs/scope_and_claim_safety.md)
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md)

---

## Roadmap

- Chuẩn hóa end-to-end evaluation runner để giảm phụ thuộc vào đường dẫn cục bộ.
- Mở rộng benchmark trên paired synthetic data và qualitative real-photo set.
- Cải thiện automatic repair mask và giảm false positive ngoài vùng hư hại.
- Bổ sung curated visual gallery cho color restoration và optional face restoration.
- Mở rộng kiểm thử runtime cho external LaMa và CodeFormer integration.

---

## Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [Architecture](docs/architecture.md) | Kiến trúc kỹ thuật và trách nhiệm module |
| [Restoration Evaluation](docs/restoration_evaluation.md) | Kết quả paired restoration evaluation |
| [Experiment Summary](docs/experiment_summary.md) | Lịch sử thí nghiệm và quyết định thiết kế |
| [Evaluation Protocol](docs/evaluation_protocol.md) | Phạm vi metric và evaluation claim |
| [Reproducibility](docs/reproducibility.md) | Artifact và minimal replay path |
| [Artifact Packaging](docs/artifacts.md) | Quy tắc đóng gói artifact |
| [External Dependencies](docs/external_dependencies.md) | LaMa, CodeFormer và checkpoint policy |
| [Deployment](docs/deployment.md) | Local và Docker deployment template |
| [Limitations](docs/limitations.md) | Giới hạn hiện tại |
| [Scope and Claim-Safety](docs/scope_and_claim_safety.md) | Ranh giới mô tả và công bố kết quả |

---

## Đóng góp

Khi mở issue hoặc pull request:

1. Mô tả rõ module, input, expected behavior và actual behavior.
2. Không commit dataset, checkpoint, external model source tree hoặc generated output lớn.
3. Chạy `python -m pytest -q` cho thay đổi liên quan đến code.
4. Chạy `python scripts/verify_artifacts.py check-all --repo-root .` nếu thay đổi manifest hoặc artifact policy.
5. Cập nhật README/docs khi thay đổi CLI, output contract, metric hoặc claim boundary.
6. Giữ các mô tả về LaMa, CodeFormer và benchmark đúng với phạm vi đã được kiểm chứng.

---

## Ghi nhận và giấy phép

Dự án sử dụng:

- [LaMa](https://github.com/advimman/lama) làm pretrained inpainting backend.
- [CodeFormer](https://github.com/sczhou/CodeFormer) cho optional face restoration.
- PyTorch, OpenCV, Gradio, Pillow, PyYAML và Kornia trong implementation.

Repository được phát triển cho mục đích học thuật trong môn **Deep Learning**. Repository hiện chưa cung cấp một file `LICENSE` độc lập; trước khi tái phân phối hoặc sử dụng ngoài phạm vi học thuật, cần kiểm tra giấy phép của source code, external dependency, pretrained weight, dataset và artifact liên quan.
