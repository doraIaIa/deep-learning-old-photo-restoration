# Demo Script

## Mục tiêu trình bày

- Trình bày repo submission như một pipeline mô-đun đang vận hành ở phạm vi `Module 1 + hybrid mask + official/pretrained LaMa`.
- Tránh mô tả các phần chưa hoàn chỉnh như thể đã là capability chính thức của repo.

## Kịch bản 3-5 phút

1. Nói rõ phạm vi tái lập hiện tại:
   `R013_REPRO` là checkpoint tham chiếu vận hành cho Module 1.
2. Nói rõ dữ liệu `R013`:
   tập ban đầu có `120` ảnh nhưng chỉ có `118` cặp ảnh-mask hợp lệ trong `masks_fixed`.
3. Nhấn mạnh threshold chính dùng cho báo cáo và fair comparison là `0.50`.
4. Giới thiệu pipeline chính:
   segmentation -> hybrid mask -> `repair_wide_v1` -> official/pretrained LaMa.
5. Nêu rõ Module 3 hiện là optional/prototype, không phải flow bắt buộc của submission.

## Câu nói an toàn khi bảo vệ

- Vì sao dùng pipeline mô-đun:
  tách riêng segmentation, mask refinement và inpainting giúp quan sát trung gian tốt hơn và tránh overclaim một mô hình end-to-end hoàn chỉnh khi evidence hiện tại mạnh nhất nằm ở Module 1.
- Vì sao dùng hybrid mask:
  mask học sâu và mask heuristic cổ điển bù trừ nhau trước bước refine.
- Vì sao không commit checkpoint vào Git:
  repo submission chỉ giữ code, config template và tài liệu; checkpoint được tham chiếu qua external path hoặc manifest.
- Vì sao không claim LaMa fine-tune:
  repo hiện dùng official/pretrained LaMa qua wrapper external runtime.
- Vì sao không claim LPIPS/FID:
  các metric này chưa có artifact hoàn chỉnh trong submission hiện tại.

## Điều không nên nói

- Không nói `R013` có `120 valid pairs`.
- Không nói `R012` là cải tiến cuối cùng.
- Không nói `LaMa` đã fine-tune.
- Không nói `CodeFormer` đã bảo toàn danh tính.
- Không nói repo đã có full quantitative end-to-end evaluation.
