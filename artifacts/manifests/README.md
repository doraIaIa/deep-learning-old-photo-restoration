# Artifact Manifests

Thư mục này dùng để mô tả artifact local và artifact ngoài repo theo dạng manifest, thay vì commit binary lớn hoặc full dataset vào Git.

## Nguyên tắc

- Checkpoint binary không commit vào Git theo mặc định.
- Full dataset không commit vào Git theo mặc định.
- Manifest giữ vai trò mô tả:
  - artifact nào đang có local;
  - artifact nào chỉ tồn tại ở external source;
  - checksum/SHA256 nào được kỳ vọng;
  - artifact nào chỉ mang tính historical.

## Cách đọc `status`

- `available`: file hiện có thể tồn tại trong local workspace ở đúng path được khai báo.
- `available_external_only`: artifact có external source rõ ràng nhưng không yêu cầu phải nằm trong repo workspace.
- `available_local_only`: artifact local tồn tại như runtime snapshot/visual support, không phải source-of-truth training data.
- `not_found_or_historical_only`: không còn đủ artifact local để kiểm chứng trực tiếp; row chỉ giữ vai trò historical/provenance.

## Vì sao không commit binary

- Giữ repo submission nhẹ và dễ review.
- Tránh biến repo submission thành kho chứa artifact research.
- Tách source code khỏi checkpoint, dataset, outputs và logs.

## Kiểm tra artifact local

Dùng:

```bash
python -B scripts/verify_artifacts.py --help
```

Ví dụ:

```bash
python -B scripts/verify_artifacts.py check-checkpoints --repo-root .
python -B scripts/verify_artifacts.py check-datasets --repo-root .
python -B scripts/verify_artifacts.py check-all --repo-root . --strict
```

Script chỉ kiểm tra manifest, path, Git tracking và checksum. Script không tự tải, không tự copy và không sửa repo.
