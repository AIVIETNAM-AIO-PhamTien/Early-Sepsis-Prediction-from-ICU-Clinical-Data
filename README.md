# Early Sepsis Prediction from ICU Clinical Data

Hệ thống **cảnh báo sớm sepsis (early warning system)** cho bệnh nhân ICU — dự đoán nguy cơ sepsis **trước 6
giờ** so với thời điểm khởi phát lâm sàng, dựa trên dữ liệu ICU theo thời gian (PhysioNet/CinC Challenge 2019).

Để có model đối chiếu ban đầu, project reproduce lại giải pháp của team **Sepsyd** (paper "Automated Prediction
of Sepsis Onset Using Gradient Boosted Decision Trees", CinC 2019) — đây là **một bước** trong lộ trình, không
phải mục tiêu cuối. Trên nền model đó, project xây thêm pipeline retraining/serving theo hướng production:
ingestion dữ liệu theo lô, chuẩn hoá qua kiến trúc medallion (bronze/silver/gold), huấn luyện lại theo lịch qua
Airflow, model registry, và một ứng dụng Streamlit để xem dự đoán kèm diễn giải bằng ngôn ngữ tự nhiên.

Chi tiết đầy đủ về kiến trúc, kết quả, và hạn chế kỹ thuật: xem [`docs/REPORT.md`](docs/REPORT.md).

## Kiến trúc hệ thống

```
data/incoming ──▶ bronze ──▶ silver ──▶ quality gate ──▶ gold
                                                            │
                                                            ▼
                                        Airflow DAG (dags/sepsis_retraining_dag.py)
                                cross-validate ─▶ train ─▶ evaluate ─▶ register ─▶ promote
                                                            │
                                                            ▼
                                artifacts/current_model.json  (model registry, con trỏ atomic)
                                                            │
                              ┌─────────────────────────────┴─────────────────────────────┐
                              ▼                                                            ▼
                    src/inference (ModelLoader → predictor →                    app/streamlit_app.py
                    pipeline "sepsyd" hoặc "team_v1")                           (upload, predict, watchlist,
                              │                                                  model monitoring)
                              ▼
                    src/narrative (diễn giải kết quả bằng Gemini,
                    fallback template nếu không có API key)
```

## Cấu trúc thư mục

```
.
├── docs/                    # Báo cáo, dataset overview, đề xuất kiến trúc
├── notebooks/               # eda_v2_pipeline_aligned.ipynb (hiện hành) + archive/ + reproduction/
├── vendor/sepsyd_original/  # Code inference gốc của team Sepsyd (vendored, read-only)
├── src/
│   ├── pipelines/           # ingestion, processing (cleaning/feature_engineering/preprocessor),
│   │                        # training, validation, registry, observability — dùng chung cho Airflow DAG
│   ├── inference/           # loader, predictor, 2 pipeline (sepsyd/team_v1), CLI, monitoring
│   ├── narrative/           # diễn giải kết quả bằng LLM
│   └── shared/              # paths, schema dùng chung
├── app/                     # Streamlit UI
├── dags/                    # Airflow DAG retraining
├── configs/                 # retraining.yaml, data_schema.json
├── data/                    # bronze/silver/gold/raw/incoming/quarantine/registry (không commit dữ liệu thật)
├── artifacts/               # model registry (current_model.json + models/<version>/)
├── scripts/                 # bootstrap, convert model legacy
└── tests/                   # pytest
```

## Cài đặt

```bash
pip install -e .
# hoặc, không cần cài package:
pip install -r requirements.txt
```

## Dữ liệu

Dữ liệu training (PhysioNet/CinC Challenge 2019) đóng gói sẵn trên Kaggle:
**https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data**

Mô tả đầy đủ features, format file, cách chấm điểm utility score: [`docs/DATASET_OVERVIEW.md`](docs/DATASET_OVERVIEW.md).

Đặt file `.psv` vào `data/raw/` — chấp nhận cả cấu trúc phẳng lẫn lồng thư mục Set A/B (xem
[`data/raw/README.md`](data/raw/README.md)).

## Sử dụng

### 1. Bootstrap model artifacts (lần đầu)

```bash
python scripts/bootstrap_artifacts.py
```

Script xác minh model Sepsyd portable và tạo `artifacts/models/v1/` + `artifacts/current_model.json`. Model mặc
định lưu bằng XGBoost JSON. Nếu cần dựng lại JSON từ pickle 0.90 gốc, khởi động Docker rồi chạy:

```bash
bash scripts/convert_legacy_sepsyd_model.sh
```

### 2. Test inference offline (CLI)

```bash
python -m src.inference.cli --input app/sample_data/p000001.psv
python -m src.inference.cli --input app/sample_data/p000001.psv --output out.csv
```

### 3. Chạy Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Upload file `.psv` / `.csv`, hoặc chọn file mẫu trong `app/sample_data/`, rồi nhấn **Run Prediction**.

Sau khi predict xong, app tự gọi LLM (Gemini free tier) để diễn giải kết quả bằng tiếng Việt. Cấu hình key một
lần trong `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"   # https://aistudio.google.com/apikey
```

Không có key hợp lệ → tự dùng template nội bộ.

## Airflow retraining

Khung retraining production nằm trong `src/pipelines/`, DAG tại `dags/sepsis_retraining_dag.py`, cấu hình tại
`configs/retraining.yaml`. Đặt batch mới cần retrain vào `data/incoming/`, cài project ở chế độ editable
(`pip install -e .`) rồi khởi động Airflow.

Sau mỗi lần retrain thành công, candidate được đăng ký và luôn thay thế model hiện tại trong
`current_model.json` (chưa so sánh performance với model cũ — xem hạn chế bên dưới). Ứng dụng và DAG dùng chung
`artifacts/current_model.json`; app tự reload ở lần dự đoán tiếp theo sau khi DAG promote.

Chi tiết đầy đủ: [`docs/REPORT.md`](docs/REPORT.md) mục 4.

## Kiểm thử

```bash
pytest tests/ -v
```

## Tài liệu liên quan

- [`docs/REPORT.md`](docs/REPORT.md) — kiến trúc chi tiết, hiện trạng, hạn chế kỹ thuật, hướng phát triển.
- [`docs/DATASET_OVERVIEW.md`](docs/DATASET_OVERVIEW.md) — mô tả dataset PhysioNet Challenge 2019.
- [`docs/PRODUCTION_STRUCTURE_PROPOSAL.md`](docs/PRODUCTION_STRUCTURE_PROPOSAL.md) — hồ sơ đề xuất/thực thi tái cấu trúc thư mục.
- [`notebooks/README.md`](notebooks/README.md) — notebook nào hiện hành, notebook nào archive/reproduction.

## Hạn chế hiện tại

DAG retraining chưa so sánh candidate với model cũ trước khi promote; pipeline `team_v1` chưa từng chạy retrain
thật trong repo này; còn vài dependency/test chưa dọn xong. Danh sách đầy đủ: [`docs/REPORT.md`](docs/REPORT.md) mục 8.

## Phân công thực hiện

| Vị trí | Phụ trách báo cáo | Deliverable kỹ thuật | Definition of Done |
|---|---|---|---|
| Leader | Mục 1, 2, 4, 8, 9; review toàn bộ | Scope, architecture, risk register, release plan, tích hợp các nhánh | Các thuật ngữ/metric/version nhất quán; không merge khi gate fail |
| Data | Mục 3.1 | Data contract, ingestion, EDA, split manifest, preprocessing/feature package | Schema + leakage tests pass; data card và snapshot hash đầy đủ |
| Model | Mục 3.2 và phần model trong Mục 5 | Baseline, XGBoost, tuning, calibration, threshold, SHAP, model card | Reproducible run; evaluation artifact; không chạm test trước khi freeze |
| QA/QC | Mục 6.1, 7 | Streamlit UI, timeline/explanation, error handling, audit hooks | Contract/UI tests pass; demo đủ valid/invalid/low-quality cases |
| Pipeline | Mục 6.2 | Monitoring, retrain orchestration, registry, champion–candidate gate, canary/rollback | Dry-run và rollback drill pass; artifact/version lineage đầy đủ |
| Cả nhóm | Review chéo và demo | README, video, presentation, incident drill | Hai người review mỗi PR có ảnh hưởng data/model/safety |
