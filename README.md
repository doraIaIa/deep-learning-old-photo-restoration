<div align="center">

# Phục Hồi Ảnh Cũ với Deep Learning

### Old Photo Restoration - Modular Deep Learning Pipeline

**Crack Segmentation · Attention Gate U-Net · Hybrid Mask Construction · Pretrained LaMa Inpainting**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-f97316?style=for-the-badge)](http://127.0.0.1:7860)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Template-2496ed?style=for-the-badge&logo=docker&logoColor=white)](docs/deployment.md)

<br>

<img src="docs/assets/demo3/contact_sheet.png" alt="Old Photo Restoration Demo" width="930"/>

<sub><b>Input photo</b> &nbsp;→&nbsp; <b>Predicted repair mask</b> &nbsp;→&nbsp; <b>LaMa restored output</b></sub>

</div>

---

## 1. Mô tả ngắn

Dự án này tập trung vào bài toán phục chế ảnh cũ (old photo restoration), cụ thể là các hư hại có cấu trúc (structured damage) như vết xước, nứt, hoặc rách nhỏ. Thay vì sử dụng một mô hình end-to-end duy nhất, dự án áp dụng kiến trúc **modular restoration pipeline**. Pipeline này minh họa quá trình phục chế ảnh cũ với trọng tâm là phát hiện và xử lý vùng hư hại thông qua phân vùng vùng hư hại (damage segmentation) và điền khuyết ảnh (inpainting).

---

## 2. Pipeline chính

Kiến trúc pipeline ưu tiên tính minh bạch bằng cách xuất ra các kết quả trung gian để dễ dàng đánh giá:

```text
      Input old photo (RGB)
               │
               ▼
┌─────────────────────────────┐
│  R013 U-Net + Attention Gate│  ← Deep Learning segmentation
│  CrackSegmenter             │    phát hiện vùng cần sửa
└──────────────┬──────────────┘
               │  dl_mask.png
               ▼
┌─────────────────────────────┐
│  Classical CV Crack Detector│  ← CLAHE + morphology + component filter
│  bắt crack mảnh tương phản  │    bổ sung cho DL mask
└──────────────┬──────────────┘
               │  cv_mask.png
               ▼
┌─────────────────────────────┐
│  Hybrid Union Mask          │  union_mask = max(dl_mask, cv_mask)
└──────────────┬──────────────┘
               │  union_before_refine.png
               ▼
┌─────────────────────────────┐
│  repair_wide_v1 Refinement  │  tinh chỉnh mặt nạ lai (hybrid mask refinement)
│  làm mask phù hợp cho LaMa  │
└──────────────┬──────────────┘
               │  final_mask.png
               ▼
┌─────────────────────────────┐
│  Pretrained LaMa Wrapper    │  external inpainting runtime
└──────────────┬──────────────┘
               │
               ▼
        Restored image (RGB)
        metadata.json
```

---

## 3. Điểm nổi bật của repo

- **Phân vùng vùng hư hại**: Blueprint considered a ResNet-34 encoder, while the final R013 implementation uses a custom Attention U-Net encoder without ResNet. Nó kết hợp Attention Gate (`CrackSegmenter`) để tập trung feature vào vùng hư hỏng, bỏ qua background nhiễu.
- **Tinh chỉnh mặt nạ lai**: Kết hợp dự đoán của Deep Learning với phương pháp Classical CV để tránh bỏ sót các nét nứt mảnh, tạo ra mặt nạ phân vùng (segmentation mask) phù hợp hơn cho bước inpainting.
- **Điền khuyết ảnh**: Gọi LaMa thông qua external inpainting backend để tái tạo hình ảnh từ mask một cách tự nhiên.
- **Tập trung quan sát**: Mọi khâu đều lưu lại các file trung gian (`dl_mask.png`, `cv_mask.png`, `union_before_refine.png`, `final_mask.png`), phục vụ việc chẩn đoán lỗi dễ dàng.

---

## 4. Demo3 controlled case study

Dự án cung cấp một ca minh họa có kiểm soát demo3 (controlled demo3 case study) dùng để kiểm thử (smoke/regression test). 

Các artifact demo3 là bằng chứng định tính trên một ca minh họa có kiểm soát, không phải benchmark cấp tập dữ liệu (not a dataset-level benchmark).

| Stage                |                      Metric |   Result |
| -------------------- | --------------------------: | -------: |
| Mask-bypass pipeline |      MAE vs golden restored |      0.0 |
| Mask-bypass pipeline | Max diff vs golden restored |        0 |
| Auto-mask final mask |          IoU vs golden mask |   0.9998 |
| Auto-mask final mask |             Mask area ratio |   0.0980 |
| Auto-mask restored   |     PSNR vs golden restored | 66.64 dB |

> These are regression metrics for the fixed `demo3` case. They are not a benchmark over the full real old-photo domain.

### Demo gallery

<p align="center">
  <img src="docs/assets/demo3/input.png" width="280" alt="Input old photo"/>
   
  <img src="docs/assets/demo3/final_mask.png" width="280" alt="Final repair mask"/>
   
  <img src="docs/assets/demo3/restored_before_face.png" width="280" alt="Restored output"/>
</p>
<p align="center">
  <sub>Input image  ·  Final repair mask  ·  Restored output</sub>
</p>

<p align="center">
  <img src="docs/assets/demo3/overlay_mask.png" width="620" alt="Mask overlay"/>
</p>

---

## 5. Checkpoints and metrics

### Operational checkpoint R013

| Thông tin       | Chi tiết                                                            |
| ---------------- | -------------------------------------------------------------------- |
| Canonical folder | `checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/`  |
| Checkpoint file  | `best_val_iou.pth`                                                 |
| Load key         | `model_state_dict`                                                 |
| SHA256           | `a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725` |
| Role             | Current operational segmentation checkpoint                          |

Checkpoint binaries are not committed to Git. They must be supplied locally or through an external artifact store.

### Training summary

Mô hình segmentation R013 được fine-tune từ R011 trên một tập dữ liệu nhỏ gồm **118 valid image-mask pairs**.

| Metric    | R011 baseline | R013 selected checkpoint |
| --------- | ------------: | -----------------------: |
| IoU       |        0.2527 |         **0.3457** |
| F1        |        0.4025 |         **0.5097** |
| Precision |        0.4112 |         **0.5887** |
| Recall    |        0.4083 |                   0.4670 |
| Val IoU   |             - |                   0.3812 |
| Val F1    |             - |                   0.5503 |

> These are segmentation/fair-comparison metrics for Module 1. They are not full end-to-end restoration benchmark scores.

---

## 6. Cách chạy nhanh

### 6.1. Clone repo

```bash
git clone https://github.com/doraIaIa/deep-learning-old-photo-restoration.git
cd deep-learning-old-photo-restoration
```

### 6.2. Tạo môi trường Python

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

### 6.3. Cấu hình external dependencies

Windows:

```bash
copy configs\external_paths.example.yaml configs\external_paths.yaml
```

Linux / macOS:

```bash
cp configs/external_paths.example.yaml configs/external_paths.yaml
```

Chỉnh `configs/external_paths.yaml` để trỏ tới LaMa runtime và optional CodeFormer runtime trên máy local.

### 6.4. Đặt checkpoint R013

Expected local path:

```text
checkpoints/segmenter/seg-unet-attn-r013-gen120-fixed118-local/best_val_iou.pth
```

Expected SHA256:

```text
a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725
```

### 6.5. Kiểm tra readiness

```bash
python scripts/check_readiness.py
```

Strict mode:

```bash
python scripts/check_readiness.py --strict
```

### 6.6. Chạy pipeline

**Auto-mask mode (Windows)**:

```bash
python scripts\run_pipeline.py ^
  --image examples\inputs\demo3.png ^
  --output-dir examples\outputs\demo3_auto ^
  --face-mode off ^
  --reference examples\golden\demo3_r013_repair_wide\restored_before_face.png ^
  --reference-mask examples\golden\demo3_r013_repair_wide\final_mask.png
```

**Auto-mask mode (Linux / macOS)**:

```bash
python scripts/run_pipeline.py \
  --image examples/inputs/demo3.png \
  --output-dir examples/outputs/demo3_auto \
  --face-mode off \
  --reference examples/golden/demo3_r013_repair_wide/restored_before_face.png \
  --reference-mask examples/golden/demo3_r013_repair_wide/final_mask.png
```

*Lưu ý*: Mặc định pipeline sẽ chạy với segmenter `r013_custom_attnunet` (được khuyên dùng do kết quả inpainting tốt nhất). Bạn có thể thử nghiệm với ResNet-34 encoder bằng cách truyền cờ `--segmenter-arch r014_resnet34` (lưu ý model này cho mask mỏng hơn và có thể giảm chất lượng PSNR).

**Gradio demo**:

```bash
python scripts/run_gradio_demo.py
```

Open `http://127.0.0.1:7860`.

---

## 7. Cấu trúc repo

| Script                       | Role                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `build_dataset.py`         | Prepare or validate dataset metadata according to the manifest-driven layout |
| `build_demo_assets.py`     | Prepare demo assets used by documentation                                    |
| `check_readiness.py`       | Check local artifacts, configuration and dependency readiness                |
| `download_checkpoints.py`  | Document or assist external checkpoint retrieval where applicable            |
| `evaluate_segmentation.py` | Segmentation evaluation utility                                              |
| `run_ablation.py`          | Ablation/status utility for documented experiment variants                   |
| `run_gradio_demo.py`       | Launch the local Gradio demo                                                 |
| `run_pipeline.py`          | Run the restoration pipeline on an input image                               |
| `smoke_lama_inpainting.py` | Smoke test for the LaMa wrapper                                              |
| `train_r013_finetune.py`   | R013-specific fine-tuning/reproduction entrypoint                            |
| `train_segmentation.py`    | General segmentation training entrypoint                                     |
| `verify_artifacts.py`      | Validate artifact metadata and local checkpoint availability                 |

```text
deep-learning-old-photo-restoration/
├── app/
│   └── gradio_demo.py
├── artifacts/
│   └── manifests/
├── checkpoints/
│   └── segmenter/
├── configs/
│   ├── checkpoints.yaml
│   ├── external_paths.example.yaml
│   └── inference.yaml
├── data/
│   ├── manifests/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── docs/
│   ├── assets/demo3/
│   ├── deployment.md
│   ├── demo_script.md
│   ├── experiment_summary.md
│   ├── external_dependencies.md
│   ├── limitations.md
│   ├── reproducibility.md
│   └── scope_and_claim_safety.md
├── examples/
│   ├── golden/demo3_r013_repair_wide/
│   └── inputs/demo3.png
├── old_photo_restoration/
│   └── __init__.py
├── scripts/
├── src/old_photo_restoration/
│   ├── config.py
│   ├── pipeline.py
│   ├── evaluation/
│   ├── face_restoration/
│   ├── inpainting/
│   ├── segmentation/
│   └── utils/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 8. Claim-safety / Scope boundary

Để giữ tính minh bạch và tránh các đánh giá quá mức (overclaim), dự án áp dụng các giới hạn an toàn sau:

- **LaMa**: LaMa được dùng như pretrained/external inpainting backend; repo không claim fine-tune LaMa. Không có sự tinh chỉnh nào đối với trọng số của kiến trúc inpainting này.
- **CodeFormer**: CodeFormer là thành phần tùy chọn; repo không đảm bảo bảo toàn danh tính (no identity-preservation guarantee). Pipeline chính và kết quả đánh giá phân vùng (segmentation) không phụ thuộc vào CodeFormer.
- **Benchmark & Metrics**: LPIPS/FID không được claim như kết quả định lượng hoàn chỉnh trong repo hiện tại.
- **Demo3 Case Study**: Các artifact demo3 là bằng chứng định tính trên một ca minh họa có kiểm soát, không phải benchmark cấp tập dữ liệu. Không chứng minh khái quát cho toàn bộ miền dữ liệu ảnh cũ.

Xem chi tiết trong [`docs/limitations.md`](docs/limitations.md) và [`docs/scope_and_claim_safety.md`](docs/scope_and_claim_safety.md).

---

## 9. Reproducibility notes

Golden regression case:

```text
examples/golden/demo3_r013_repair_wide/
├── final_mask.png
├── metadata.json
└── restored_before_face.png
```

Rebuild README/demo assets:

```bash
python scripts/build_demo_assets.py
```

Create a clean zip archive without local ignored files:

```bash
git archive --format=zip --output old_photo_restoration_release.zip main
```

SHA256 values for golden artifacts are documented in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## 10. External Dependencies & Acknowledgements

This project uses [LaMa](https://github.com/advimman/lama) as the inpainting backend and keeps an optional integration path for [CodeFormer](https://github.com/sczhou/CodeFormer). When extending or redistributing external repositories, pretrained weights, or derived artifacts, follow the original licenses and citation requirements.

Repo này được phát triển cho mục đích học thuật trong môn **Deep Learning**.

---

## License

Academic course project. Check the licenses of external dependencies and pretrained weights before redistribution or use outside the academic setting.
