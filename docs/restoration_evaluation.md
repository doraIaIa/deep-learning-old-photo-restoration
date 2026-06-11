# Báo cáo Đánh giá Phục hồi (Restoration Evaluation)

Tài liệu này trình bày các kết quả đánh giá định lượng (quantitative evaluation) cho 파ipline phục hồi ảnh cũ, nhằm xác định chất lượng inpainting và nhận diện nguyên nhân gây ra lỗi (bottleneck) trong quá trình phục hồi.

## 1. Mục đích (Purpose)

Đánh giá phục hồi định lượng là cần thiết để vượt qua những hạn chế của việc chỉ kiểm tra bằng mắt thường (qualitative demo). Tuy nhiên, vì ảnh cũ thực tế thường không có ảnh gốc hoàn hảo (clean ground truth) để đối chiếu, chúng tôi bắt buộc phải sử dụng dữ liệu tổn thương tổng hợp có cặp (synthetic paired data). Phép đánh giá này giúp đo lường một cách khách quan khả năng khôi phục cấu trúc và màu sắc tại vùng bị hỏng.

## 2. Dữ liệu và Thiết lập Đánh giá (Dataset and Evaluation Setup)

- **Dataset:** Tập dữ liệu tổng hợp `ds-crack3d` có cặp (synthetic paired data).
- **Clean source:** Các ảnh gốc sạch được lấy từ tập DIV2K.
- **Cấu trúc:** Gồm ảnh bị làm hỏng (degraded images), mặt nạ vùng hỏng (damage masks) và ảnh gốc sạch (clean ground truth).
- **Backend:** V3 pipeline kết hợp với mô hình LaMa cục bộ đã được pretrained (pretrained local LaMa).
- **Lưu ý:**
  - Không thực hiện fine-tuning (huấn luyện bổ sung) trên mô hình LaMa.
  - Đây **không phải** là benchmark trên ảnh cũ thực tế (not real-photo benchmark).

## 3. Thang đo Đánh giá (Metrics)

Các thang đo được sử dụng:
- **PSNR (Peak Signal-to-Noise Ratio):** Chỉ số cao hơn (Higher is better) cho thấy chất lượng ảnh khôi phục gần với ảnh gốc hơn.
- **MAE (Mean Absolute Error):** Chỉ số thấp hơn (Lower is better) cho thấy sai lệch ít hơn.
- **Masked-region MAE:** Tính MAE chỉ tính riêng trong phạm vi các pixel bị đánh dấu hỏng bởi mặt nạ (mask), giúp cô lập đánh giá chất lượng inpainting thuần túy.
- *(Lưu ý: Chỉ số SSIM hiện không khả dụng trong môi trường hiện tại và không được tính toán. Cả PSNR và MAE được chia làm hai loại: đánh giá trên toàn bộ khung hình và đánh giá cục bộ tại vùng masked).*

## 4. Kết quả Pipeline Tự động (Auto/hybrid pipeline result, n=30)

| Metric | Degraded baseline | Auto/hybrid restored | Delta | Improved samples |
|---|---|---|---|---|
| Full-image PSNR | 19.127 | 17.355 | -1.772 | 0/30 |
| Full-image MAE | 24.984 | 27.989 | +3.005 | 0/30 |
| Masked-region MAE | 35.666 | 34.151 | -1.515 | 15/30 |

**Nhận xét (Interpretation):**
- Thang đo trên toàn ảnh (full-image metrics) suy giảm khi sử dụng chế độ tự động sinh mặt nạ (auto/hybrid mode).
- Tuy nhiên, độ lỗi tại vùng tổn thương thực tế (Masked-region MAE) lại được cải thiện trên 15/30 mẫu.
- Điều này cho thấy có tồn tại tín hiệu sửa chữa cục bộ (local repair signal) tích cực, nhưng tổng thể hành vi của toàn pipeline tự động vẫn chưa ổn định.

## 5. Phân tách Lỗi với Oracle-mask (Oracle-mask ablation, n=15)

Đánh giá này truyền thẳng ground-truth mask vào mô hình LaMa để kiểm chứng khả năng của riêng backend inpainting.

| Metric | Degraded baseline | Auto/hybrid | Oracle-mask LaMa | Interpretation |
|---|---|---|---|---|
| Full-image PSNR | 17.974 | 16.698 | 17.991 | oracle slightly improves over degraded and improves over auto/hybrid |
| Full-image MAE | 27.739 | 30.316 | 27.712 | oracle slightly improves over degraded and improves over auto/hybrid |
| Masked-region MAE | 35.554 | 33.983 | 31.318 | oracle improves masked-region error most clearly |
| PSNR improved count | — | 0/15 | 10/15 | oracle near-majority but below 70% threshold |
| Masked MAE improved count | — | 5/15 | 10/15 | oracle stronger than auto/hybrid |

**Nhận xét (Interpretation):**
- Kết quả từ Oracle-mask giúp cô lập chính xác ảnh hưởng của chất lượng mặt nạ.
- Rõ ràng chất lượng mặt nạ tự động (automatic mask quality) chính là điểm thắt cổ chai (bottleneck) lớn nhất.
- Do tỉ lệ cải thiện là 10/15 (đạt 66.7%), nó không vượt qua ngưỡng đa số 70% cứng.

## 6. Hạn chế (Limitations)

- Chỉ thực hiện đánh giá trên bộ dữ liệu cặp tổng hợp (synthetic paired ds-crack3d only).
- Đây không phải là benchmark đánh giá trên ảnh cũ thực tế.
- Bước chạy Oracle-mask chỉ mang ý nghĩa chuẩn đoán lỗi (diagnostic), không phải là thiết lập có thể triển khai thực tế (not deployable setting).
- Chưa có tính toán LPIPS/FID.
- Không có sẵn chỉ số SSIM.
- Kích thước mẫu đánh giá Oracle-mask còn nhỏ (n=15).
- Các ảnh cũ thật (real old photos) vẫn cần được đánh giá thông qua chất lượng thị giác (qualitative).

## 7. Kết luận (Report-safe takeaway)

Các kết quả định lượng cho thấy pipeline tự động chưa cải thiện chất lượng toàn ảnh trên synthetic paired data, nhưng oracle-mask ablation cho thấy chất lượng mask là bottleneck chính: khi dùng ground-truth damage mask, masked-region MAE giảm rõ hơn và PSNR cải thiện trên 10/15 mẫu.
