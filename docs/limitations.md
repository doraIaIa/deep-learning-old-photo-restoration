# Giới hạn (Limitations)

## Phạm vi Hiện tại (Current Scope)

Pipeline hiện tại tập trung vào:

- Phân vùng (segmentation) các vùng hư hại;
- Xây dựng mặt nạ lai (hybrid mask construction);
- Sử dụng chiến lược `repair_wide_v1`;
- Một official/pretrained LaMa wrapper.

## Những Điều Chưa Phải Là Safe Claim (What Is Not a Safe Claim Yet)

- Fine-tune LaMa.
- Các phép đo `LPIPS`, `FID`, và `masked-region LPIPS`. (Được lên kế hoạch cho future evaluation protocol / hiện không claim)
- Đánh giá định lượng toàn bộ end-to-end (Full quantitative end-to-end evaluation).
- Bảo toàn danh tính (identity preservation) của CodeFormer (không được bảo đảm).
- Một luồng Module 3 face restoration hoàn chỉnh.
- Xử lý chiếu sáng (Illumination handling) như một bản triển khai hoàn chỉnh trong repo hiện tại.

## Lưu ý về Dataset và Thực nghiệm (Dataset and Experiment Caveats)

- `R013` phải luôn được mô tả là xuất phát từ `120` ảnh ban đầu nhưng chỉ có `118` cặp hợp lệ (valid pairs).
- `R012` chỉ là một nhánh thực nghiệm (experimental branch) với `15` mẫu thủ công.
- `demo3` là một golden regression case, không phải là một benchmark đại diện cho toàn bộ tập ảnh cũ thực tế.

## Công việc Tương lai (Future Work)

- Fine-tune LaMa với các artifact đầy đủ.
- Đo lường LPIPS/FID/masked-region LPIPS (các giao thức đánh giá tương lai, hiện không được claim).
- Một luồng Module 3 mạnh mẽ hơn.
- Một giao thức đánh giá (evaluation protocol) hoàn thiện hơn thay vì chỉ dừng ở các smoke/regression checks.
