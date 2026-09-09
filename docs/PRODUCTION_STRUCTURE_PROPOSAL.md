# Đề xuất tái cấu trúc thư mục cho chuẩn production

> **Trạng thái: đã thực thi phần lớn (2026-09-09).** Tài liệu này được giữ lại làm hồ sơ quyết định (vấn đề gì,
> đề xuất cấu trúc nào, tại sao) — nội dung "Vấn đề hiện tại" và "Kế hoạch di chuyển" bên dưới mô tả trạng thái
> **trước khi** thực thi. Cấu trúc thực tế hiện tại xem ở `README.md` (root); các mục còn để ngỏ (đổi tên
> `src` → `src/sepsis`, tách `sigmoid`, wire `performance_gate.py` vào DAG) chưa được làm — xem
> [REPORT.md](REPORT.md) mục "Hướng phát triển tiếp theo".

Tài liệu này liệt kê các vấn đề cấu trúc hiện tại (đã khảo sát trực tiếp trong repo), đề xuất cây thư mục mục tiêu, và một kế hoạch di chuyển an toàn theo từng bước. Bối cảnh đầy đủ về hệ thống xem tại [REPORT.md](REPORT.md).

## 1. Vấn đề hiện tại (bằng chứng cụ thể)

| # | Vấn đề | Bằng chứng |
|---|---|---|
| 1 | Logic xử lý dữ liệu "thật" nằm ngoài package `src/`, phải nạp bằng `importlib` vì tên file có dấu gạch ngang | `data/cleaning.py`, `data/feature-engineering.py` được `src/pipelines/processing/preprocessor.py` và `feature_builder.py` load qua `importlib.util.spec_from_file_location` thay vì `import` chuẩn |
| 2 | Không có package cài đặt được | Không có `pyproject.toml`/`setup.py`/`setup.cfg` ở root; import chạy được nhờ `pythonpath = .` trong `pytest.ini` và `sys.path.insert(0, str(ROOT))` thủ công trong `app/streamlit_app.py`, `tests/conftest.py` |
| 3 | Notebook rời rạc, không rõ bản canonical | `model-comparison.ipynb`, `reproduce-xai.ipynb` ở repo root; `data/data-pipeline.ipynb`, `data/eda_ver1.ipynb` lẫn trong thư mục data; `reproduce/` có 3 biến thể không ghi chú |
| 4 | Code vendor (bên thứ ba) lẫn với code của project | `original/` là bản vendor verbatim từ `physionetchallenges/2019ChallengeEntries` (Python 3.7, xgboost 0.90, có `Dockerfile`/`requirements.txt` riêng xung đột tooling với phần còn lại) nhưng nằm ngang hàng với `src/`, `app/` ở root |
| 5 | Doc lệch với thực tế (doc drift) | `data/raw/README.md` trỏ `python -m scripts.bootstrap_splits` — không tồn tại trong `scripts/`; `artifacts/models/README.md` mô tả `artifacts/models/model_v1/` với `threshold.json`/`metrics.json`/`cv_results.json`/... — khác tên (`v1`) và thiếu file so với thực tế |
| 6 | Dependency không rõ mục đích | `requirements.txt` có `fastapi`, `starlette`, `svcs`, `cadwyn` nhưng không có code nào trong `src/`/`app/` dùng |
| 7 | Artifact dư thừa | `artifacts/models/v1/model.pkl` tồn tại song song `model.json` dù manifest chỉ tham chiếu `model.json` |
| 8 | Coupling serving ↔ training | `src/inference/pipelines/team_v1.py` import `src/pipelines/training/objectives.py` (chỉ để lấy `sigmoid`) — inference production phải load cả module chỉ dành cho training (CV search, custom gradient) |
| 9 | Tài liệu rời rạc, không có nơi tập trung | README con nằm rải rác (`data/raw/README.md`, `data/incoming/README.md`, `artifacts/models/README.md`, `app/sample_data/README.md`), không có `docs/` |
| 10 | Không có CI | Không có `.github/workflows/`, không Makefile, không pre-commit |

## 2. Cây thư mục đề xuất

```
.
├── pyproject.toml                     # MỚI — package "sepsis" cài editable, khai báo deps thay requirements.txt
├── README.md
├── REPORT.md
├── docs/
│   ├── DATASET_OVERVIEW.md            # chuyển từ root (giữ 1 nơi tập trung tài liệu)
│   ├── PRODUCTION_STRUCTURE_PROPOSAL.md
│   └── data-governance.md             # gộp nội dung data/raw/README.md, data/incoming/README.md, artifacts/models/README.md sau khi sửa lệch thực tế
├── src/
│   └── sepsis/                        # ĐỔI TÊN gốc import: "sepsis" thay vì "src" trần (chuẩn src-layout)
│       ├── pipelines/
│       │   ├── ingestion/
│       │   ├── processing/
│       │   │   ├── cleaning.py            # CHUYỂN từ data/cleaning.py, bỏ import qua importlib
│       │   │   ├── feature_engineering.py # CHUYỂN từ data/feature-engineering.py, bỏ dấu gạch ngang
│       │   │   ├── feature_builder.py     # sửa lại import chuẩn thay vì importlib hack
│       │   │   ├── preprocessor.py        # sửa lại import chuẩn thay vì importlib hack
│       │   │   ├── silver_pipeline.py
│       │   │   └── gold_pipeline.py
│       │   ├── training/
│       │   │   └── objectives.py          # tách sigmoid ra sepsis/shared/math.py (xem mục 3)
│       │   ├── validation/
│       │   ├── registry/
│       │   └── observability/
│       ├── inference/
│       ├── narrative/
│       └── shared/
│           └── math.py                # MỚI — sigmoid và hàm toán học dùng chung, không phụ thuộc training
├── app/                                # giữ nguyên vị trí (Streamlit)
├── dags/                               # giữ nguyên
├── configs/                            # giữ nguyên
├── data/                               # CHỈ còn data thật: bronze/silver/gold/raw/incoming/quarantine/registry
├── notebooks/                          # MỚI — gộp toàn bộ notebook rời rạc [Đã thực thi]
│   ├── archive/                        # model-comparison.ipynb, eda_ver1.ipynb, data-pipeline.ipynb, reproduce-xai.ipynb
│   └── reproduction/                   # nội dung reproduce/ (3 notebook + PDF), có README nêu rõ bản nào hiện hành
├── vendor/
│   └── sepsyd_original/                # CHUYỂN từ original/ — đánh dấu rõ read-only/tham chiếu, không sửa
├── artifacts/                          # giữ nguyên, dọn model.pkl thừa
├── scripts/                            # giữ nguyên
├── tests/                              # mirror theo src/sepsis/* thay vì phẳng
│   ├── pipelines/
│   ├── inference/
│   └── narrative/
├── .github/
│   └── workflows/
│       └── ci.yml                      # MỚI — chạy pytest (và lint nếu có) trên mỗi PR
└── logs/                               # giữ nguyên
```

**Lưu ý về mức độ mạnh tay**: đổi `src/` → `src/sepsis/` là thay đổi có ảnh hưởng rộng nhất (mọi `from src.xxx import yyy` phải đổi thành `from sepsis.xxx import yyy`). Nếu muốn giảm rủi ro, có thể làm theo 2 giai đoạn: (a) trước mắt chỉ thêm `pyproject.toml` giữ nguyên tên gói `src`, dọn `data/*.py` và notebook; (b) đổi tên `src` → `sepsis` ở một đợt riêng sau khi đã ổn định. Quyết định này nên hỏi lại trước khi thực thi.

## 3. Nợ kỹ thuật đi kèm (đề xuất, cần quyết định riêng trước khi thực thi)

- **Tách `sigmoid` dùng chung**: tạo `src/shared/math.py` (hoặc `sepsis/shared/math.py`) chỉ chứa các hàm toán học thuần (vd. `sigmoid`), để `team_v1.py` (inference) không phải import `training/objectives.py` (vốn có cả custom gradient objective chỉ dùng lúc train).
- **Dependency thừa** (`fastapi`, `starlette`, `svcs`, `cadwyn`): cần hỏi rõ ý định ban đầu — nếu là kế hoạch xây API service riêng (thay/song song Streamlit) thì giữ lại và ghi chú rõ trong `pyproject.toml` (extra group `api`); nếu không, xoá khỏi dependency chính.
- **`artifacts/models/v1/model.pkl`**: xác nhận không còn nơi nào đọc file này (chỉ `current_model.json` trỏ `model.json`) rồi mới xoá.
- **Wire `performance_gate.py` vào DAG**: đây là thay đổi hành vi (ảnh hưởng khi nào một model được promote), không phải thuần cấu trúc thư mục — nên tách thành task riêng có review kỹ, không gộp chung với việc dọn cấu trúc thư mục.

## 4. Kế hoạch di chuyển từng bước (khi được duyệt thực thi)

Thực hiện tuần tự, chạy `pytest tests/ -v` sau **mỗi** bước để đảm bảo không hỏng gì trước khi sang bước kế tiếp:

1. **Thêm `pyproject.toml`** khai báo package hiện có (giữ nguyên tên `src`). **[Đã thực thi — quyết định thực tế: giữ nguyên `pythonpath = .` trong `pytest.ini` và các `sys.path.insert()` thay vì bỏ, vì vẫn hoạt động song song không xung đột]**
2. **Di chuyển `data/cleaning.py`, `data/feature-engineering.py`** vào `src/pipelines/processing/` (đổi tên bỏ gạch ngang), sửa import trong `preprocessor.py` và `feature_builder.py` từ `importlib.util.spec_from_file_location` sang `import` chuẩn. **[Đã thực thi]**
3. **Tạo `notebooks/`**, chuyển `model-comparison.ipynb`, `reproduce-xai.ipynb` (root) và `data/data-pipeline.ipynb`, `data/eda_ver1.ipynb` vào `notebooks/archive/`; chuyển `reproduce/` thành `notebooks/reproduction/`, viết README nêu rõ notebook nào là bản hiện hành. **[Đã thực thi]**
4. **Chuyển `original/` → `vendor/sepsyd_original/`**, cập nhật đường dẫn tham chiếu trong `README.md`, `REPORT.md`, `src/inference/pipelines/sepsyd.py`, `scripts/convert_legacy_sepsyd_model.sh`. **[Đã thực thi]**
5. **Cập nhật config/DAG/README** nếu path nào bị ảnh hưởng — không cần sửa gì (`data/bronze|silver|gold|...`, `configs/retraining.yaml`, `dags/`, `src/shared/paths.py` không đổi path). **[Đã xác nhận]**
6. **Dọn artifact/dependency**: gỡ `model.pkl` thừa khỏi git tracking (giữ file trên máy), xoá 4 dependency không dùng (`fastapi`/`starlette`/`svcs`/`cadwyn`) khỏi `requirements.txt`.
7. **Gộp docs**: tạo `docs/`, chuyển `DATASET_OVERVIEW.md`/`REPORT.md` vào đó, sửa nội dung lệch thực tế trong `data/raw/README.md` và `artifacts/models/README.md`.
8. **Thêm CI**: `.github/workflows/ci.yml` chạy `pytest tests/ -v` trên push/PR. **[Chưa làm — nằm ngoài phạm vi đợt này]**
9. **(Tuỳ chọn, tách riêng)** Đổi `src` → `src/sepsis` nếu quyết định theo đúng src-layout chuẩn — cần sed/rename toàn bộ import, nên làm ở một PR riêng, review kỹ.
10. **(Tuỳ chọn, tách riêng)** Tách `sigmoid` ra `shared/math.py` và wire `performance_gate.py` vào DAG — đây là thay đổi hành vi, không thuộc phạm vi "dọn cấu trúc thư mục" thuần tuý.

## 5. Rủi ro khi thực thi

- **Airflow cache DAG theo path file** — sau khi đổi cấu trúc import trong `src/pipelines/`, cần restart Airflow scheduler/webserver để nạp lại DAG, tránh chạy nhầm code cũ đã cache.
- **`src/shared/paths.py` dùng env var override** (`SEPSIS_ARTIFACTS_DIR`, ...) nên tương đối an toàn với việc đổi vị trí thư mục, nhưng vẫn cần rà lại toàn bộ nơi dùng path cứng dạng string (`"data/bronze"`, `"artifacts/models"`, ...) rải rác trong `configs/retraining.yaml` và code, để đảm bảo không có path bị hardcode sai chỗ khác.
- **Notebook đã chạy sẵn (đã có output/cell đã lưu)** — khi di chuyển file `.ipynb`, đường dẫn tương đối bên trong notebook (đọc `../data/...`) có thể cần sửa lại.
- Nên thực thi qua từng PR nhỏ (theo đúng thứ tự ở mục 4) thay vì một lần đổi tất cả, để dễ review và dễ revert nếu có bước nào gây lỗi.
