Quá trình xây dựng mặt nạ lai bằng các phép hình thái học (dilate, repair, close).

Hybrid mask construction process.

Minh chứng cách mà nhiều heuristics kết hợp lại để tạo ra một mặt nạ vững chắc (Demonstrates how multiple heuristics combine to form a solid mask).
Điều này quan trọng để chứng minh sự ổn định của đường ống phục chế (Matters to prove the pipeline's robustness).

Nguồn gốc (Provenance):
Được tái tạo lại từ repo phát triển trên dữ liệu đầu vào demo3 có kiểm soát (Reproduced from the development repository on the controlled demo3 input).

Lưu ý (Caveat):
Đây là các dữ liệu định tính có kiểm soát (Qualitative controlled-case evidence), không phải là một benchmark trên toàn bộ tập dữ liệu (not a dataset-level benchmark).

An toàn Claim (Claim safety):
- Repo không claim fine-tune LaMa (No LaMa fine-tuning is claimed).
- Không claim LPIPS/FID cho artifact này (LPIPS/FID are not claimed for this artifact).
- Không bảo đảm giữ danh tính gốc (No identity-preservation guarantee is claimed).
