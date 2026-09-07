# AIO-Microwave-Sepsis

Reproduce baseline từ paper "Automated Prediction of Sepsis Onset Using Gradient Boosted Decision Trees" (team Sepsyd, CinC 2019) — dự đoán sớm sepsis từ dữ liệu lâm sàng ICU, PhysioNet/CinC Challenge 2019.

## Hiện trạng project

- **`original/`** — mã nguồn **inference gốc của chính tác giả** (team Sepsyd), tải từ repo chính thức `physionetchallenges/2019ChallengeEntries`. Gồm `get_sepsis_score.py` (thuật toán inference), model XGBoost đã train sẵn (`f120d4e02n8010val434.pickle.dat`), `driver.py`, paper gốc kèm theo. Lưu ý: đây **chỉ có code inference, không có code training** — tác giả không công khai phần này.
- **`app/`** — Streamlit UI upload + dự đoán sepsis (MVP).
- **`src/inference/`** — Module inference: validator, model loader, pipeline sepsyd, prediction log.
- **`artifacts/`** — Model v1 + `current_model.json` manifest.
- **`reproduce/reproduce_sepsis_baseline.ipynb`** — nơi đang thực hiện việc **reproduce lại baseline** theo mô tả thuật toán trong paper (preprocessing, feature engineering, train XGBoost, đánh giá bằng utility score chính thức của Challenge).
- **`reproduce/PublishedPaperCinC2019-423.pdf`** — bản paper gốc.
- **`DATASET_OVERVIEW.md`** — tài liệu mô tả dataset PhysioNet Challenge 2019.
- **`requirements.txt`** — dependency cho reproduce + app (numpy, pandas, scikit-learn, xgboost, streamlit, pytest).

## Dataset

Dữ liệu training (PhysioNet/CinC Challenge 2019) được đóng gói sẵn trên Kaggle:

**https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data**

Xem chi tiết mô tả features, format file, và cách chấm điểm utility score trong [`DATASET_OVERVIEW.md`](DATASET_OVERVIEW.md).

## Chạy ứng dụng dự đoán

### 1. Cài dependency

```bash
pip install -r requirements.txt
```

### 2. Bootstrap model artifacts (lần đầu)

```bash
python scripts/bootstrap_artifacts.py
```

Script tải model Sepsyd gốc (nếu chưa có trong `original/`) và tạo `artifacts/models/v1/` + `artifacts/current_model.json`.

### 3. Test inference offline (CLI)

```bash
python -m src.inference.cli --input app/sample_data/p000001.psv
python -m src.inference.cli --input app/sample_data/p000001.psv --output out.csv
```

### 4. Chạy Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Upload file `.psv` / `.csv`, hoặc chọn file mẫu trong `app/sample_data/`, rồi nhấn **Run Prediction**.

### Tường thuật LLM (tự động sau khi predict)

Sau khi predict xong, app **tự gọi LLM** (Gemini free tier) để diễn giải kết quả bằng tiếng Việt.

Cấu hình key một lần trong `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"   # https://aistudio.google.com/apikey
```

Không có key hợp lệ → tự dùng template nội bộ. Không cần cấu hình gì trên UI.

### 5. Chạy test

```bash
pytest tests/ -v
```

## Trạng thái hiện tại

- MVP prediction app với pipeline `sepsyd` (model v1 gốc).
- Pipeline `team_v1` (model ML mới) — stub, chờ artifact v2 từ nhóm retrain.
- Đã có nguồn dữ liệu training trên Kaggle — đang chạy `reproduce/reproduce_sepsis_baseline.ipynb` để reproduce baseline.

## Airflow retraining

Khung retraining production nằm trong `src/pipelines/`, DAG tại `dags/sepsis_retraining_dag.py` và cấu hình tại `configs/retraining.yaml`.

Chuẩn bị asset trước khi chạy:

1. Đặt toàn bộ file `.psv` trực tiếp vào `data/raw/`.
2. Đặt batch mới cần retrain vào `data/incoming/`.
3. Cài dependency bằng `pip install -r requirements.txt`, cấu hình `PYTHONPATH` trỏ tới project root rồi khởi động Airflow.

DAG tạo lại train/test split theo patient ở mỗi lần retrain và lưu split metadata trong thư mục run. Sau mỗi lần retrain thành công, candidate được đăng ký và luôn thay thế model hiện tại trong `current_model.json`, không qua bước so sánh performance với model cũ.

### Chạy kiểm thử

```bash
python -m pip install -r requirements.txt
pytest
```

Test suite kiểm tra DAG import/topology trên Airflow 3, batch detection và Bronze ingestion, data quality gate, lookback theo từng bệnh nhân, performance gate, dataset registry và model promotion. Test không chạy huấn luyện XGBoost hoàn chỉnh nên có thể chạy nhanh trong quá trình phát triển.
