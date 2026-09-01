# AIO-Microwave-Sepsis

Reproduce baseline từ paper "Automated Prediction of Sepsis Onset Using Gradient Boosted Decision Trees" (team Sepsyd, CinC 2019) — dự đoán sớm sepsis từ dữ liệu lâm sàng ICU, PhysioNet/CinC Challenge 2019.

## Hiện trạng project

- **`original/`** — mã nguồn **inference gốc của chính tác giả** (team Sepsyd), tải từ repo chính thức `physionetchallenges/2019ChallengeEntries`. Gồm `get_sepsis_score.py` (thuật toán inference), model XGBoost đã train sẵn (`f120d4e02n8010val434.pickle.dat`), `driver.py`, paper gốc kèm theo. Lưu ý: đây **chỉ có code inference, không có code training** — tác giả không công khai phần này.
- **`reproduce/reproduce_sepsis_baseline.ipynb`** — nơi đang thực hiện việc **reproduce lại baseline** theo mô tả thuật toán trong paper (preprocessing, feature engineering, train XGBoost, đánh giá bằng utility score chính thức của Challenge).
- **`reproduce/PublishedPaperCinC2019-423.pdf`** — bản paper gốc.
- **`DATASET_OVERVIEW.md`** — tài liệu mô tả dataset PhysioNet Challenge 2019.
- **`requirements.txt`** — dependency cho phần reproduce (numpy, pandas, scikit-learn, xgboost).

## Dataset

Dữ liệu training (PhysioNet/CinC Challenge 2019) được đóng gói sẵn trên Kaggle:

**https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data**

Xem chi tiết mô tả features, format file, và cách chấm điểm utility score trong [`DATASET_OVERVIEW.md`](DATASET_OVERVIEW.md).

## Trạng thái hiện tại

Đã có nguồn dữ liệu training trên Kaggle — đang chạy `reproduce/reproduce_sepsis_baseline.ipynb` để reproduce baseline.

## Airflow retraining

Khung retraining production nằm trong `src/pipelines/`, DAG tại `dags/sepsis_retraining_dag.py` và cấu hình tại `configs/retraining.yaml`.

Chuẩn bị asset trước khi chạy:

1. Đặt toàn bộ file `.psv` trực tiếp vào `data/raw/`.
2. Đặt batch mới cần retrain vào `data/incoming/`.
3. Cài dependency bằng `pip install -r requirements.txt`, cấu hình `PYTHONPATH` trỏ tới project root rồi khởi động Airflow.

DAG tạo lại train/test split theo patient ở mỗi lần retrain và lưu split metadata trong thư mục run. Sau mỗi lần retrain thành công, candidate được đăng ký và luôn thay thế model hiện tại trong `current_model.json`, không qua bước so sánh performance với model cũ.
