# Ca minh họa có kiểm soát Demo3 (Demo3 Controlled Case Study)

Thư mục này chứa một bộ nhỏ các bản sao chẩn đoán đầu ra đã được chắt lọc (curated set of reproduced diagnostic outputs) trên một dữ liệu đầu vào (input) demo3 duy nhất và có kiểm soát. Mục đích là để giải thích quá trình tiến hóa của đường ống phục chế (pipeline evolution) và các quyết định thiết kế thông qua việc sử dụng một bức ảnh nhất quán qua các giai đoạn: tạo mặt nạ (mask generation), tinh chỉnh mặt nạ (mask refinement), và đầu ra phục chế cuối cùng (final restoration output).

Các artifact này là các dữ liệu định tính có kiểm soát (qualitative controlled-case evidence), không phải là một benchmark trên toàn bộ tập dữ liệu (not a dataset-level benchmark). Thư mục 03 bị thiếu do không có ảnh so sánh được chắt lọc công khai (no public curated comparison image was available).

## Các Giai đoạn (Stages)

- 01_input_and_problem
- 02_auto_mask_or_pipeline_output
- 04_hybrid_mask_construction
- 05_final_pipeline_output
