# Data Policy

Repo submission không chứa full dataset huấn luyện trong Git.

## External datasets chính

- `F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq`
- `F:\deeplearning\r013_finetune_set-20260608T080800Z-3-001\r013_finetune_set`

## Repo này giữ gì

- manifest dữ liệu mức tóm tắt;
- sample/golden nhỏ nếu cần cho demo hoặc docs;
- có thể giữ split nhỏ hoặc manifest checksum trong phase sau nếu thật sự cần cho reproducibility.

## Repo này không giữ gì

- full images/masks của dataset huấn luyện;
- manual-mask workspace lớn;
- overlays/quality checks đầy đủ;
- runtime outputs lớn.

## Thiết lập path

- path external được cấu hình theo máy qua `configs/external_paths.yaml` hoặc manifest liên quan;
- không commit full dataset chỉ để làm repo trông “đầy đủ”.

## Mục đích của thư mục `data/`

Thư mục `data/` trong submission repo, nếu được dùng, nên là nơi chứa:

- README chính sách dữ liệu;
- manifest nhỏ;
- split snapshot nhỏ;
- sample tối thiểu.

Không dùng thư mục này để copy dataset thật vào Git.
