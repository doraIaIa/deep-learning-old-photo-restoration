# Tổng hợp Thực nghiệm (Experiment Summary)

Dự án này không gộp toàn bộ lịch sử nghiên cứu vào repo. Tài liệu này lưu giữ các điểm quan trọng nhằm bảo đảm claim safety và khả năng tái tạo tối thiểu (minimal reproducibility).

## Tóm tắt Module 1

- Checkpoint reference tái tạo mạnh nhất là `R013_REPRO`.
- `R013` bắt đầu từ `120` ảnh nhưng chỉ có `118` valid image-mask pairs trong `masks_fixed`.
- Các mục `masks_fixed` bị thiếu là `real_0099` và `real_0112`.
- Tập chia cố định (fixed split) là `83 / 18 / 17`.
- `R013_REPRO` được khởi tạo từ `R011_REPRO`, không phải từ `R012_REPRO`.
- Ngưỡng đánh giá (reporting threshold) chính là `0.50`.
- IoU tập validation của `R013` lịch sử là `0.381231`; IoU validation khi tái tạo (reproduced validation IoU) gần bằng, ở mức `0.380532`.
- Đánh giá công bằng (fair test) cho `R013_REPRO @0.50`: IoU/F1 = `0.337970 / 0.501339`.
- Đánh giá công bằng cho `R011_REPRO @0.55`: IoU/F1 = `0.246848 / 0.394876`.
- Mức tăng trưởng qua so sánh công bằng: `+0.091122 IoU`, `+0.106463 F1`.

## Trạng thái R012

- `R012` là một nhánh thực nghiệm với tập con manual-subset gồm `15` mẫu (samples).
- `R012` không phải là bản cải tiến cuối cùng.
- `R012` không được dùng làm checkpoint khởi tạo (initialization checkpoint) cho `R013`.

## Tóm tắt Module 2

- Repo hiện tại sử dụng bản official/pretrained LaMa thông qua một external runtime wrapper.
- Repo không claim fine-tune LaMa.
- Các loss như `L1`, `perceptual`, và `adversarial` nên được xem là công việc tương lai (future work) trừ khi có sẵn các artifact fine-tune.

## Giới hạn Đánh giá (Evaluation Boundary)

- Có minh chứng mạnh mẽ (strong evidence) cho các segmentation metrics của Module 1 như `IoU`, `F1`, `Precision`, và `Recall`.
- Repo không claim đã hoàn thiện đo lường (completed) `LPIPS`, `FID`, hoặc `masked-region LPIPS`.
- `demo3` là một golden regression case dùng cho smoke/demo checks, không phải là một benchmark đánh giá toàn bộ trên tập ảnh cũ thực tế.

## Minh chứng Vận hành (Operational Evidence) được lưu trong Repo

- Repo giữ các smoke/golden artifacts cho `demo3` tại `seg_smoke_demo3`, `pipeline_smoke_demo3`, và `gradio_smoke_demo3`.
- Các artifact này hữu ích cho operational checks và regression inspection.
- Chúng không nên được diễn giải là các điểm số đánh giá toàn bộ (full end-to-end quantitative evaluation).

## Minh chứng Tối thiểu (Minimal Evidence) được lưu trong Repo

- README và docs mô tả giới hạn vận hành hiện tại.
- Small golden artifacts cho `demo3`.
- Hỗ trợ CLI và Gradio demo để kiểm tra smoke/readiness checks.

## Chuyển sang Công việc Tương lai (Moved to Future Work)

- Fine-tune LaMa.
- Đánh giá LPIPS/FID/masked-region LPIPS (đã lên kế hoạch cho future work, hiện không được claim).
- Full end-to-end quantitative evaluation.
- Hoàn thiện một luồng (flow) Module 3 face restoration.


## Quá trình Huấn luyện (Training Lineage): R006-R013 Segmenter Development

| Run | Objective | Data / Label Target | Key Result | Decision |
|---|---|---|---|---|
| **R006** | Baseline (synthetic) | Synthetic (50 ep) | Val IoU: 0.3852, Val F1: 0.5249 (thr 0.25) | Recall weak; move to augmentation |
| **R007** | Strong augmentation | Synthetic aug | Val IoU: 0.3912, Val F1: 0.5257 (thr 0.20) | Precision improved; recall weak; change loss |
| **R008** | BCE + Tversky loss | Synthetic aug | Val IoU: 0.4064, Val F1: 0.5492 (thr 0.70) | Recall improved; extend training |
| **R009** | Synthetic pretrain (60 ep) | Synthetic aug | Val IoU: 0.4171, Val F1: 0.5595. **Real Test IoU: 0.0022** | Severe domain gap on real photos; use as base |
| **R010** | Real-domain fine-tune | Real (thin masks) | Real Test IoU: 0.2927, Test F1: 0.4528 (thr 0.70) | Domain gap overcome; masks too thin for LaMa |
| **R011** | Repair mask fine-tune | Real (repair masks) | Test IoU: 0.4478, Test F1: 0.6186 | Stable baseline; missed extremely thin cracks |
| **R012** | Manual mask constraint | Manual (15 samples) | Test IoU: 0.2846, Test F1: 0.4430 | Overfit/small-data negative experiment |
| **R013** | Operational segmenter | Fixed 118 pairs | Val F1: 0.5502, Test IoU: 0.3456 (thr 0.50) | Selected operational checkpoint (seg-unet-attn-r013-gen120-fixed118-local) |

## Quyết định Thiết kế dựa trên Lỗi (Failure-Driven Design Decisions)

- **Modular Pipeline**: Các nỗ lực khôi phục end-to-end trực tiếp ban đầu đã tạo động lực cho việc phân rã bài toán thành pipeline modular (segmentation và inpainting) nhằm giảm regression bias.
- **Inpainting Dependency**: LaMa được sử dụng hoàn toàn dưới dạng một pretrained wrapper subprocess. Repo không claim fine-tune LaMa để tránh chuỗi huấn luyện sinh tự động không ổn định.

## Sự tiến hóa Dữ liệu Huấn luyện (Training Data Evolution)

- **Initial Datasets Rejected**: Các tập dữ liệu tương tự CrackForest hoặc ds-crack3d-512-n0200-v001 bị từ chối do không khớp về mask area (ví dụ: vết nứt nhựa đường dày hơn vết xước ảnh).
- **Synthetic Pretraining Data**: Áp dụng Crack Bank RGBA assets kết hợp với 3D degradation vật lý, normal maps, Phong illumination, và alpha blending trên nền DIV2K.
- **Domain Gap & Fine-tuning**: Domain gap (khoảng cách miền dữ liệu) nghiêm trọng được quan sát thấy ở R009 (IoU giảm xuống 0.0022 trên tập real test) đòi hỏi một chuỗi fine-tune trên miền dữ liệu thực tế (R010, R011, R013) sử dụng real photographs.


## Chiến lược Mask và Ngưỡng (Mask and Threshold Strategy)

- **Loss Progression**: Ban đầu dùng BCE+Dice. Để phạt nặng false negatives, R008 giới thiệu Tversky Loss (alpha=0.3, beta=0.7). R011 tăng beta=0.8 cho các repair masks hướng tới recall.
- **Threshold Evolution**: Inference thresholds (ngưỡng suy luận) biến thiên linh hoạt dựa trên phân bố độ tự tin của mô hình (ví dụ: R007 tại 0.20, R009 tại 0.90). Mô hình segmenter R013 cuối cùng sử dụng một operational threshold ổn định là 0.50.
- **Hybrid Mask Refinement**: Deep learning mask được kết hợp (union) với một classical CV branch (CLAHE, Blackhat, Canny). Chiến lược `repair_wide_v1` sau đó áp dụng morphological closing, connection, và dilation để chuẩn bị final mask cho bước inpainting.
