# Raw bootstrap data

Đặt toàn bộ dữ liệu PhysioNet/CinC 2019 trực tiếp tại đây:

```text
data/raw/
├── p000001.psv
├── p000002.psv
├── ...
├── p100001.psv
└── p100002.psv
```

Pipeline coi tất cả file là một cohort thống nhất; không chia Set A/B và không cần thư mục con. Thư mục này dùng để tạo fixed test split và đánh giá candidate/current model. Không commit dữ liệu lên Git.

Sau khi thêm dữ liệu và cài dependency, tạo split đúng một lần:

```bash
python -m scripts.bootstrap_splits
```
