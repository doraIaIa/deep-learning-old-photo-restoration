# Experiment Summary

Repo submission không mang toàn bộ lịch sử thí nghiệm của repo research. Tài liệu này chỉ giữ các kết quả cần cho reproducibility của submission.

## Phase 2 Demo3 Result

Golden regression case hiện tại là `demo3`, không phải benchmark toàn bộ dataset.

- `final_mask_ratio`: `0.0979878066233506`
- `final_mask IoU vs golden`: `0.9997728216844948`
- restored `MAE vs golden`: `0.008635014295578003`
- restored `PSNR vs golden`: `66.63675683358014`

## Phase 1C Regression

- mask-bypass regression `MAE = 0`
- `max_diff = 0`
- `PSNR = inf`

## Ghi Chú

- Checkpoint r013 không nằm trong Git, chỉ tồn tại local.
- Kết quả trên dùng demo3 làm golden case để kiểm tra hồi quy và khả năng tái lập.
