# Evaluation Protocol

## Phạm vi hiện tại

Submission hiện có bằng chứng mạnh nhất cho evaluation của Module 1 segmentation và cho smoke/regression ở `demo3`.
Tài liệu này mô tả evaluation mong muốn và các bằng chứng đang có, không có nghĩa repo submission đã đóng gói đầy đủ mọi runner đánh giá.

## Safe Metrics

- `IoU`
- `F1`
- `Precision`
- `Recall`

Các metric trên là phần có evidence mạnh nhất trong chuỗi `R010_REPRO -> R013_REPRO`, với `R013_REPRO` là checkpoint tái lập mạnh nhất hiện tại.

## Cách mô tả an toàn

- Threshold chính để trình bày và fair comparison cho `R013` là `0.50`.
- `R013` phải được ghi là `120` ảnh ban đầu nhưng chỉ `118` valid pairs.
- `demo3` chỉ là golden regression case cho smoke/demo.

## Chưa nên claim hoàn tất

- `LPIPS`
- `FID`
- `masked-region LPIPS`
- full quantitative end-to-end evaluation

Các mục này chỉ nên được mô tả là planned evaluation hoặc future work nếu chưa có artifact rõ ràng trong repo submission.

## Runner Caveat

- Segmentation metrics hiện có evidence mạnh từ artifact và summary đã audit.
- Repo submission hiện chưa đóng gói đầy đủ runner cho toàn bộ `LPIPS`, `FID`, oracle-mask protocol và full ablation.
- `scripts/evaluate_segmentation.py` và `scripts/run_ablation.py` vẫn là mục sẽ xử lý ở Phase 1B nếu tiếp tục.
- Vì vậy không nên diễn giải docs hiện tại như bằng chứng rằng toàn bộ evaluation stack đã hoàn chỉnh.
