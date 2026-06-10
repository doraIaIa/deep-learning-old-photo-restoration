# Experiment Summary

Repo submission này không đóng gói toàn bộ lịch sử thực nghiệm của repo research cũ. Tài liệu này chỉ giữ những điểm cần cho claim safety và reproducibility tối thiểu.

## Module 1 Summary

- Checkpoint tái lập có bằng chứng mạnh nhất hiện tại là `R013_REPRO`.
- `R013` xuất phát từ `120` ảnh nhưng chỉ có `118` cặp ảnh-mask hợp lệ trong `masks_fixed`.
- Hai ảnh thiếu `masks_fixed` là `real_0099` và `real_0112`.
- Split cố định là `83 / 18 / 17`.
- `R013_REPRO` init từ `R011_REPRO`, không init từ `R012_REPRO`.
- Threshold chính để báo cáo và fair comparison là `0.50`.
- Historical val IoU của `R013` là `0.381231`; repro val IoU gần khớp ở `0.380532`.
- Fair test `R013_REPRO @0.50`: IoU/F1 = `0.337970 / 0.501339`.
- Fair test `R011_REPRO @0.55`: IoU/F1 = `0.246848 / 0.394876`.
- Delta fair comparison: `+0.091122 IoU`, `+0.106463 F1`.

## R012 Status

- `R012` là nhánh thực nghiệm/manual subset với `15` samples.
- `R012` không phải final improvement.
- `R012` không được dùng làm checkpoint khởi tạo cho `R013`.

## Module 2 Summary

- Repo submission hiện dùng official/pretrained LaMa qua wrapper external runtime.
- Không claim LaMa đã fine-tune.
- Các loss `L1`, `perceptual`, `adversarial` chỉ nên xem là future work nếu chưa có artifact fine-tune rõ ràng.

## Evaluation Boundary

- Có bằng chứng mạnh cho segmentation metrics của Module 1 như `IoU`, `F1`, `Precision`, `Recall`.
- Không claim rằng submission hiện đã hoàn tất `LPIPS`, `FID` hoặc `masked-region LPIPS`.
- `demo3` là golden regression case cho smoke/demo, không phải benchmark đầy đủ trên tập ảnh cũ thực.

## Operational Evidence Kept In Submission

- Repo hiện có smoke/golden artifacts cho `demo3` ở các nhánh `seg_smoke_demo3`, `pipeline_smoke_demo3`, `gradio_smoke_demo3`.
- Các artifact này hữu ích để kiểm tra hành vi vận hành và regression tối thiểu.
- Chúng không nên được diễn giải như full quantitative end-to-end evaluation trên tập ảnh cũ thực.

## Minimal Evidence Kept In Submission

- README và docs mô tả phạm vi vận hành hiện tại.
- Golden artifacts nhỏ cho `demo3`.
- CLI và Gradio demo để phục vụ smoke/readiness.

## Moved To Future Work

- Fine-tune LaMa.
- Đánh giá LPIPS/FID/masked-region LPIPS.
- Full quantitative end-to-end evaluation.
- Module 3 face restoration hoàn chỉnh trong flow submission.
