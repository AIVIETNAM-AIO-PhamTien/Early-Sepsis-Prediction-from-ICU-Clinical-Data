# Report: Early Sepsis Prediction from ICU Clinical Data

> Báo cáo tổng hợp về project — vừa phục vụ mục đích nộp bài (AIO2026 – Module 3), vừa là tài liệu kỹ thuật mô tả kiến trúc hệ thống hiện tại.

## 1. Tổng quan bài toán

Sepsis (nhiễm trùng huyết) là tình trạng đe dọa tính mạng khi phản ứng của cơ thể với nhiễm trùng gây tổn thương mô, suy tạng hoặc tử vong — tại Mỹ ước tính ~1.7 triệu ca/năm và ~270,000 tử vong; toàn cầu ~30 triệu ca và ~6 triệu tử vong mỗi năm. Phát hiện sớm có ý nghĩa sống còn vì mỗi giờ trì hoãn điều trị làm tăng tỷ lệ tử vong.

Project reproduce lại bài toán của **PhysioNet/Computing in Cardiology (CinC) Challenge 2019**: dự đoán sepsis **trước 6 giờ** so với thời điểm khởi phát lâm sàng chính thức (theo tiêu chí Sepsis-3: kết hợp nghi ngờ nhiễm trùng `t_suspicion` và suy giảm ≥2 điểm SOFA `t_SOFA`, nhãn dương được dịch sớm 6 giờ trước `t_sepsis`). Baseline được chọn để reproduce là giải pháp của team **Sepsyd** (paper "Automated Prediction of Sepsis Onset Using Gradient Boosted Decision Trees", CinC 2019).

Ngoài việc reproduce baseline, project còn tự xây thêm một **pipeline retraining và serving theo hướng production**: ingestion dữ liệu theo lô, chuẩn hoá qua kiến trúc medallion (bronze/silver/gold), huấn luyện lại theo lịch qua Airflow, model registry tự chế, và một ứng dụng Streamlit để bác sĩ/điều dưỡng upload dữ liệu bệnh nhân và xem dự đoán kèm diễn giải bằng ngôn ngữ tự nhiên.

## 2. Dữ liệu

Nguồn: PhysioNet/CinC Challenge 2019 (DOI `10.13026/v64v-d857`), đóng gói lại trên Kaggle (`nguyenhoangthaotrinh/sepsyd-data`). Chi tiết đầy đủ nằm ở [DATASET_OVERVIEW.md](DATASET_OVERVIEW.md); tóm tắt:

- **40,336 bệnh nhân** từ 2 hệ thống bệnh viện (Training Set A: 20,336, Set B: 20,000), mỗi bệnh nhân một file `.psv` (pipe-separated), mỗi hàng là 1 giờ đo trong ICU.
- **41 cột**: 8 sinh hiệu (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2), 26 chỉ số xét nghiệm, 6 thông tin nhân khẩu/hành chính (Age, Gender, Unit1/2, HospAdmTime, ICULOS), và nhãn `SepsisLabel`.
- Đánh giá bằng **utility score** chuyên biệt của Challenge (không dùng AUC/accuracy thuần) — thưởng dự đoán đúng trong khoảng 12–3 giờ trước `t_sepsis`, phạt nặng dự đoán muộn, phạt nhẹ dự đoán quá sớm hoặc false positive.
- **Không commit dữ liệu thật lên Git** — `.gitignore` chỉ giữ lại 3 file `.psv` mẫu dùng cho demo (`app/sample_data/`) và các `README.md`/`.gitkeep` placeholder trong cây `data/`.

### Kiến trúc dữ liệu (medallion)

Repo tự triển khai một data lake dạng medallion trong `data/` + `src/pipelines/`:

| Layer | Thư mục | Module xử lý | Vai trò |
|---|---|---|---|
| Raw | `data/raw/` | — | File `.psv` gốc, đặt phẳng (không chia Set A/B) |
| Incoming | `data/incoming/` | `src/pipelines/ingestion/batch_detector.py` | Batch mới cần nạp, phát hiện trùng lặp bằng checksum trong `data/registry/` |
| Bronze | `data/bronze/<batch_id>/` | `src/pipelines/ingestion/bronze_pipeline.py` | Copy/giải nén an toàn (chống zip-slip) từ incoming |
| Silver | `data/silver/<batch_id>/` | `src/pipelines/processing/silver_pipeline.py` | Chuẩn hoá tên cột theo alias (`configs/data_schema.json`), kiểm tra ID bệnh nhân, tính đơn điệu `ICULOS`, loại outlier tuyệt đối → `patients.parquet` + `silver_report.json` |
| Quality gate | — | `src/pipelines/validation/quality_gate.py` | Chặn dữ liệu không đạt chuẩn trước khi vào gold (thiếu cột, sai domain nhãn, trùng giờ đo, missingness bất thường) |
| Gold | `data/gold/<dataset_version>/` | `src/pipelines/processing/gold_pipeline.py` | Merge batch mới với tập "development" cũ theo bệnh nhân (replace-on-overlap) → `features.parquet`, `patient_manifest.parquet`, kèm metadata/checksum đăng ký ở `src/pipelines/registry/dataset_registry.py` |
| Quarantine | `data/quarantine/` | — | Nơi cách ly batch không đạt quality gate |

## 3. Kiến trúc hệ thống tổng thể

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

Toàn bộ hệ thống dùng chung một **model registry tự chế bằng file JSON** thay vì công cụ như MLflow: `artifacts/current_model.json` là con trỏ (ghi atomic qua temp-file + `os.replace`) trỏ tới một thư mục version trong `artifacts/models/<version>/`. Cả Airflow DAG (bên ghi) và Streamlit app/CLI (bên đọc) đều thao tác trên cùng file này, nên chỉ cần promote xong là app tự nhận model mới ở lần dự đoán tiếp theo (loader cache theo mtime).

## 4. Retraining pipeline (Airflow)

- DAG: `dags/sepsis_retraining_dag.py`, viết bằng Airflow 3 TaskFlow API (`airflow.sdk.dag/task`), lịch chạy hàng tháng (`0 2 1 * *` theo `configs/retraining.yaml`), `max_active_runs: 1`, 1 lần retry.
- Các task nối tiếp: `validate_bootstrap_assets → detect_new_batch → bronze → silver → quality → gold → cross_validate → train → evaluate → register_and_promote`. Không có batch mới hoặc batch trùng → skip qua `AirflowSkipException`.
- **Cross-validation**: `src/pipelines/training/cv_search.py`, chia theo bệnh nhân, 3-fold khi search hyperparameter (`search_n_splits`) và 10-fold khi finalize (`finalization_n_splits`), chọn theo AUPRC, tie-break bằng utility score, dò 3 tổ hợp `max_depth`/`eta` trong config.
- **Huấn luyện cuối**: `src/pipelines/training/trainer.py` — fit preprocessor, dựng ma trận lookback (`lookback_hours: 5`, `moving_window_hours: 6`, `delta_lag_hours: 1`), train `xgb.Booster` với **custom weighted-logloss objective** (`positive_weight: 40.0`, xử lý mất cân bằng nhãn nặng — sepsis là nhãn hiếm) định nghĩa ở `src/pipelines/training/objectives.py`.
- **Chọn ngưỡng quyết định**: dò từ 0.05 đến 0.95 (bước 0.05), chọn theo utility score (`src/pipelines/training/utility.py` implement công thức utility chính thức của Challenge).
- **Đánh giá & đăng ký**: `evaluator.py` tính metric trên tập test giữ lại; `src/pipelines/registry/model_registry.py` kiểm tra đủ bộ artifact bắt buộc (`REQUIRED_ARTIFACTS`) trước khi copy atomically vào `artifacts/models/<version>/`, rồi `promote_model` ghi đè `current_model.json`.
- **Quan trọng — hành vi promote hiện tại**: DAG **luôn promote candidate mới** miễn là vượt qua data-quality checks và train/evaluate thành công, **không so sánh performance với model đang chạy**. Module `src/pipelines/validation/performance_gate.py` đã tồn tại và có test riêng (`tests/test_performance_gate.py`) nhưng **chưa được gọi trong DAG** — xem thêm mục 8.
- Mọi lần chạy (thành công hay thất bại) được ghi vào `logs/retraining_history.jsonl` qua `src/pipelines/observability/retraining_logger.py`.

## 5. Inference & serving

Có **hai pipeline inference song song**, lựa chọn động theo `pipeline` field trong `current_model.json`:

- **`sepsyd`** (`src/inference/pipelines/sepsyd.py`): bản port thủ công từ thuật toán gốc của tác giả (`vendor/sepsyd_original/get_sepsis_score.py`, vendored từ repo chính thức PhysioNet challenge). Đây là model baseline được convert từ pickle XGBoost 0.90 sang **portable XGBoost JSON** (qua `scripts/convert_legacy_sepsyd_model.py`/`.sh`, chạy trong môi trường Docker cô lập vì pickle cũ không load được trên XGBoost hiện đại).
- **`team_v1`** (`src/inference/pipelines/team_v1.py`): dùng artifact do chính DAG retraining ở trên tạo ra (preprocessor + feature lookback + threshold đã học), tái sử dụng trực tiếp `src/pipelines/processing/*` và `sigmoid` từ `src/pipelines/training/objectives.py`.

Thành phần dùng chung cho cả hai: `src/inference/loader.py` (đọc manifest, cache theo mtime, load `xgb.Booster` từ JSON/UBJ), `predictor.py` (điều phối qua `PIPELINE_REGISTRY`), `validator.py` (kiểm tra schema/cột/tính đơn điệu của file upload), `io.py` (đọc `.psv`/`.csv`), `logger.py` (ghi log dự đoán JSONL theo ngày vào `logs/predictions/`), `monitoring.py` (tổng hợp log cho trang giám sát), `escalation.py` (map mức rủi ro sang hành động/khoa phụ trách — có ghi chú rõ đây là gợi ý, không phải khuyến nghị lâm sàng chính thức).

Ba cách gọi cùng một pipeline này:
1. **CLI**: `python -m src.inference.cli --input <file.psv>` — dự đoán offline.
2. **Streamlit app** (`app/streamlit_app.py`): 3 trang — `1_predict.py` (upload + dự đoán), `2_unit_watchlist.py` (theo dõi theo khoa/đơn vị), `3_model_monitoring.py` (đọc log dự đoán để giám sát). Sidebar hiển thị model/version/threshold đang active qua `ModelLoader`.
3. **Tầng diễn giải LLM** (`src/narrative/`): sau khi predict, app tự gọi Gemini (free tier) để tóm tắt kết quả bằng tiếng Việt; nếu không cấu hình `GEMINI_API_KEY` trong `.streamlit/secrets.toml`, tự động fallback sang template nội bộ (`TemplateProvider`) — không chặn luồng dự đoán chính.

## 6. Kiểm thử

Bộ test dùng **pytest** (`pytest.ini`: `testpaths = tests`, `pythonpath = .`), 10 file trong `tests/` + fixtures dùng chung ở `conftest.py`:

- `test_dag.py` — kiểm tra DAG import được trên Airflow 3, đúng topology tuyến tính, đúng retry policy (không cần Airflow server chạy thật).
- `test_ingestion.py` — batch detection, dedup theo checksum, chống zip-slip khi giải nén bronze.
- `test_feature_builder.py` — lookback không được vượt biên bệnh nhân.
- `test_quality_gate.py` — các case pass/fail của quality gate.
- `test_performance_gate.py` — logic so sánh candidate vs. baseline (module tồn tại nhưng chưa wire vào DAG — xem mục 8).
- `test_registry.py` — versioning/checksum của dataset registry, register + promote model.
- `test_validator.py` — validate input lúc inference.
- `test_prediction.py` — end-to-end dự đoán qua `ModelLoader`/`run_prediction`, kiểm tra lỗi khi thiếu artifact.
- `test_team_v1_pipeline.py` — model được promote bởi pipeline `team_v1` chạy inference đúng.
- `test_retraining_inference_contract.py` — test hợp đồng đầy đủ: train → evaluate → register → promote → predict, đảm bảo artifact DAG tạo ra tương thích với tầng inference.
- `test_narrative.py` — sinh narrative/template.

Chưa có test cho: `app/` (Streamlit pages/components), `src/inference/cli.py`, `monitoring.py`, `logger.py`, `escalation.py`, và phần lớn nội bộ `src/narrative/` (ngoài `test_narrative.py`).

## 7. Hiện trạng & kết quả

- Model đang phục vụ thật sự trong repo (`artifacts/current_model.json`): `model_version: "v1"`, `pipeline: "sepsyd"`, `model_format: "xgboost_json"`, `threshold: 0.5` — đây là **model gốc của Sepsyd đã convert format**, không phải model do DAG retraining huấn luyện.
- **Pipeline `team_v1` và toàn bộ DAG retraining đã hoàn chỉnh về code và có test hợp đồng đầy đủ, nhưng chưa từng chạy thật để promote một model mới trong repo này** — thư mục `artifacts/models/v1/` hiện chỉ có `model.json`, `model.pkl` (dư thừa), `preprocessor.json`, `feature_config.json`, thiếu hẳn `metrics.json`/`cv_results.json`/`threshold.json` mà một candidate do DAG tạo ra sẽ có (theo đúng thiết kế trong `model_registry.py`). Vì vậy **báo cáo này không nêu số liệu AUPRC/utility cụ thể** của model `team_v1` — số liệu đó chỉ tồn tại sau khi chạy DAG trên dữ liệu thật.
- Bước thiết lập model baseline đối chiếu (`notebooks/reproduction/reproduce_sepsis_baseline.ipynb`, reproduce lại theo đúng mô tả trong paper) — một hạng mục trong lộ trình xây hệ thống cảnh báo, hiện **đang thực hiện**, chưa hoàn tất/chốt số liệu.
- Các tham số huấn luyện đã cấu hình sẵn cho lần retrain tiếp theo (`configs/retraining.yaml`): split 80/20 theo bệnh nhân, positive_weight = 40 cho weighted-logloss, dò threshold 0.05→0.95, chọn theo utility score — tức phần "khung" production đã sẵn sàng, chỉ còn thiếu một lần chạy thật với dữ liệu đủ lớn để có kết quả.

## 8. Hạn chế & rủi ro kỹ thuật

1. **DAG luôn promote không điều kiện.** `performance_gate.py` tồn tại, có test, nhưng không được gọi trong `dags/sepsis_retraining_dag.py` — không có cơ chế tự động chặn việc promote một model bị regression so với model đang chạy.
2. **Coupling giữa serving và training**: `src/inference/pipelines/team_v1.py` import trực tiếp từ `src/pipelines/processing/*` và `src/pipelines/training/objectives.py` (chỉ để lấy hàm `sigmoid`) — tầng inference khi chạy production phải load cả code chỉ dùng cho training (CV search, custom gradient objective).
3. **Dependency thừa/chưa rõ mục đích**: `requirements.txt` khai báo `fastapi`, `starlette`, `svcs`, `cadwyn` nhưng không có bất kỳ đoạn code nào trong `src/`/`app/` sử dụng — có thể là dự định xây API service riêng chưa triển khai, hoặc cruft cần dọn.
4. **Tài liệu lệch với thực tế (doc drift)**: `data/raw/README.md` hướng dẫn chạy `python -m scripts.bootstrap_splits` — script này không tồn tại trong `scripts/`. `artifacts/models/README.md` mô tả layout `artifacts/models/model_v1/` với đầy đủ `threshold.json`/`metrics.json`/`cv_results.json`/... — khác tên thư mục (`v1` chứ không phải `model_v1`) và thiếu các file đó so với thực tế hiện tại.
5. **Artifact thừa**: `artifacts/models/v1/model.pkl` tồn tại song song với `model.json` dù `current_model.json` chỉ tham chiếu `model.json` (`model_format: xgboost_json`) — dễ gây nhầm lẫn về format nào là chuẩn.
6. **Không có CI/pre-commit/Makefile** — mọi kiểm tra (test, có thể cả lint) hiện phải chạy thủ công.
7. **Ngôn ngữ tài liệu không nhất quán**: README/DATASET_OVERVIEW/hầu hết README con dùng tiếng Việt, trong khi toàn bộ code/docstring dùng tiếng Anh.

## 9. Hướng phát triển tiếp theo

1. Wire `performance_gate.py` vào bước `register_and_promote` của DAG để không tự động promote model bị regression.
2. Tách phần logic dùng chung tối thiểu (vd. `sigmoid`) ra khỏi `training/objectives.py` để `team_v1.py` (inference) không phải phụ thuộc code chỉ dành cho training.
3. Chạy thật một lần retrain trên dữ liệu đầy đủ để có số liệu AUPRC/utility thật cho `team_v1`, hoàn tất `notebooks/reproduction/reproduce_sepsis_baseline.ipynb` để có baseline đối chiếu.
4. Dọn dependency thừa (`fastapi`/`starlette`/`cadwyn`/`svcs`) hoặc làm rõ kế hoạch dùng chúng; xoá `model.pkl` thừa nếu xác nhận không còn cần.
5. Sửa các README bị lệch thực tế; bổ sung test cho `app/`, CLI, monitoring/logger/escalation.
6. Thêm CI (chạy `pytest tests/ -v` tối thiểu) qua GitHub Actions.

Chi tiết đề xuất cấu trúc thư mục cụ thể cho các điểm 2 và 4–5 được trình bày riêng trong [docs/PRODUCTION_STRUCTURE_PROPOSAL.md](docs/PRODUCTION_STRUCTURE_PROPOSAL.md).
