# Raw bootstrap data

Đặt dữ liệu PhysioNet/CinC 2019 vào đây, ví dụ:

```text
data/raw/
├── p000001.psv
├── p000002.psv
├── ...
├── p100001.psv
└── p100002.psv
```

Chấp nhận cả cấu trúc phẳng lẫn lồng thư mục theo Set A/B (`training_setA/`, `training_setB/`) — các notebook
đọc dữ liệu ở đây (`notebooks/eda_v2_pipeline_aligned.ipynb`) dùng `rglob("*.psv")` nên tự tìm được file dù có
lồng thư mục.

Pipeline retraining production (Airflow DAG) **không đọc trực tiếp thư mục này** — DAG đọc batch mới từ
`data/incoming/`. `data/raw/` chỉ phục vụ EDA và các notebook reproduce baseline.

Không commit dữ liệu thô lên Git.
