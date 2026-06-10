# Restoration Evaluation

> **Tóm tắt tiếng Việt**
>
> - Đánh giá khôi phục ảnh sử dụng bộ dữ liệu tổng hợp (synthetic paired data).
> - Chỉ đo lường chất lượng inpainting dựa trên LaMa (backend). LaMa hoàn toàn là mô hình pretrained bên ngoài, dự án không claim fine-tuning LaMa.
> - Đây không phải benchmark trên ảnh cũ thực tế. Mọi kết quả đo lường (như PSNR, SSIM) chỉ áp dụng cho bộ dữ liệu tổng hợp này.

## Protocol
- **Mode:** Oracle-mask (evaluate inpainting quality using ground-truth mask).
- **Sample Size:** 30
- **Metrics:** PSNR, SSIM, MAE, MSE

## Results Summary
- **Baseline PSNR:** N/A dB
- **Restored PSNR:** N/A dB

*Note: Demo3 remains qualitative controlled-case evidence.*
