# Scripts

Thư mục này chứa các lệnh công khai để chuẩn bị dữ liệu, huấn luyện, đánh giá,
chạy inference và kiểm tra artifact. Chạy các lệnh từ thư mục gốc của repository.

## Pipeline và demo

- `run_pipeline.py`: chạy toàn bộ pipeline phục hồi ảnh cũ.
- `run_color_restoration.py`: chạy riêng module phục hồi màu sau LaMa.
- `run_gradio_demo.py`: khởi chạy giao diện Gradio.
- `run_ablation.py`: chạy hoặc kiểm tra các cấu hình ablation.
- `eval_pipeline_paired.py`: đánh giá pipeline trên dữ liệu paired.

## Dữ liệu và huấn luyện

- `build_dataset.py`: chuẩn bị hoặc kiểm tra dataset theo manifest.
- `train_segmentation.py`: huấn luyện model phân đoạn vết nứt.
- `train_r013_finetune.py`: fine-tune model phân đoạn theo cấu hình R013.
- `evaluate_segmentation.py`: đánh giá model phân đoạn.

## Kiểm tra và tiện ích

- `check_readiness.py`: kiểm tra config và artifact local cần thiết.
- `verify_artifacts.py`: xác thực metadata và checkpoint.
- `download_checkpoints.py`: hỗ trợ tải checkpoint bên ngoài.
- `build_demo_assets.py`: tạo asset dùng cho demo và tài liệu.
- `smoke_lama_inpainting.py`: smoke test LaMa.
- `smoke_r014_segmenter.py`: smoke test segmenter R014.

Checkpoint, dataset, ảnh inference và output sinh ra trong runtime không được
commit lên Git. Repository chỉ theo dõi source code, config, manifest và tài liệu
cần để tái lập quy trình.
