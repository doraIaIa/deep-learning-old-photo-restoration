# Phạm vi và Lưu ý An toàn (Scope and Claim-Safety Notes)

Tài liệu này làm rõ phạm vi triển khai chính xác, các chính sách về artifact, và những giới hạn rõ ràng của dự án Old Photo Restoration. Mục đích của tài liệu là thiết lập các ranh giới không thể hiểu nhầm về những gì repo này cung cấp.

## Phạm vi Triển khai (Implemented Scope)
- **Module 1**: Crack Segmentation. Sử dụng kiến trúc Attention U-Net. Operational segmenter: R013.
- **Module 1.5**: Hybrid Mask Refinement. Kết hợp mask xác suất của deep learning với hình thái học (morphology) của computer vision cổ điển để khôi phục các vết xước mảnh.
- **Module 2**: Inpainting Backend. Sử dụng một pre-trained LaMa (Large Mask Inpainting) model wrapper.
- **Module 3**: Face Restoration tùy chọn. Cung cấp tích hợp CodeFormer + RetinaFace.

## Chính sách Checkpoint và Artifact
- **Cách ly nghiêm ngặt (Strict Isolation)**: Checkpoint binaries và datasets được giữ hoàn toàn cục bộ (strictly local). Chúng bị bỏ qua bởi version control (Git) nhằm tránh làm phình to repository. Chỉ có các file siêu dữ liệu (metadata manifests) và thư mục khung (folder skeletons) được commit.
- **Minh chứng Lịch sử (Historical Evidence)**: Các lần huấn luyện vòng lặp trước đây (ví dụ: R006-R008) được giữ lại dưới dạng khung thư mục (evidence-only skeletons).
- **Current Binaries**: Các checkpoint hiện tại (ví dụ: R009-R013) vẫn là các local ignored binaries. R013 được chỉ định làm operational segmenter hiện tại.

## Giới hạn Rõ ràng và Các Claim An toàn (Explicit Limitations and Safe Claims)
Nhằm duy trì tính toàn vẹn của dự án, các giới hạn sau đây sẽ được áp dụng:
- **Tích hợp LaMa**: LaMa inpainting backend được sử dụng hoàn toàn dưới dạng một pretrained subprocess wrapper. **Không claim fine-tune LaMa** (no fine-tuning) nào được thực hiện trong pipeline này.
- **Đánh giá Định lượng (Quantitative Metrics)**: Repo này không claim đã hoàn thiện đo lường LPIPS, FID, hay masked-region LPIPS. Chúng vẫn thuộc về các giao thức đánh giá tương lai.
- **Face Restoration**: CodeFormer là một dependency tùy chọn và **không bảo đảm giữ danh tính gốc** (no identity-preservation guarantee).
- **Historical Originals**: Các bản sao tái tạo (reproduction checkpoints) không được xem là các bản gốc lịch sử nếu không có bằng chứng xác minh (verifiable evidence).
