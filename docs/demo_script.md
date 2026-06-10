# Kịch bản Demo (Demo Script)

## Mục tiêu Trình bày (Presentation Goal)

- Trình bày project repo dưới dạng một pipeline modular (modular pipeline) hoạt động trong phạm vi `Module 1 + hybrid mask + official/pretrained LaMa`.
- Tránh mô tả các thành phần chưa hoàn thiện như thể chúng là các tính năng đã sẵn sàng cho production.

## Tiến trình 3-5 Phút Gợi ý (Suggested 3-5 Minute Flow)

1. Nêu rõ phạm vi có thể tái tạo hiện tại:
   `R013_REPRO` là tham chiếu checkpoint vận hành (operational checkpoint) cho Module 1.
2. Nêu rõ các thông tin về dataset của `R013`:
   tập dữ liệu ban đầu có `120` ảnh nhưng chỉ tồn tại `118` valid image-mask pairs trong `masks_fixed`.
3. Nhấn mạnh rằng ngưỡng chính (main threshold) được dùng để báo cáo và so sánh công bằng là `0.50`.
4. Giới thiệu pipeline cốt lõi:
   segmentation -> hybrid mask -> `repair_wide_v1` -> official/pretrained LaMa.
5. Nêu rõ rằng Module 3 hiện tại là tùy chọn/thử nghiệm (optional/prototype) và không nằm trong luồng bắt buộc.

## Các Luận điểm An toàn (Safe Talking Points)

- Tại sao dùng pipeline modular:
   việc tách biệt segmentation, mask refinement, và inpainting giúp cải thiện khả năng quan sát (observability) và tránh việc overclaim một mô hình end-to-end hoàn chỉnh khi mà minh chứng mạnh mẽ nhất hiện đang nằm ở Module 1.
- Tại sao dùng hybrid mask:
   learned mask và heuristic mask bù đắp cho nhau trước khi bước vào refinement.
- Tại sao không commit các checkpoints vào Git:
   project repo lưu trữ mã nguồn, templates, và tài liệu; các checkpoint được tham chiếu thông qua external paths hoặc manifests.
- Tại sao không claim fine-tune LaMa:
   repo hiện tại sử dụng bản official/pretrained LaMa thông qua một external runtime wrapper.
- Tại sao không claim LPIPS/FID:
   các chỉ số này chưa có đầy đủ artifact hoàn thiện trong repo hiện tại.

## Những Điều Không Nên Nói (Things Not to Say)

- Không nói `R013` có `120 valid pairs`.
- Không nói `R012` là bản cải tiến cuối cùng (final improvement).
- Không nói LaMa đã được fine-tune.
- Không nói CodeFormer giữ được danh tính gốc (preserves identity).
- Không nói repo đã có đầy đủ quantitative end-to-end evaluation.
