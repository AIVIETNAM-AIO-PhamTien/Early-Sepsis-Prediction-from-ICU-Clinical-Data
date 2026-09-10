# Notebooks

Dự án hướng tới xây dựng một **hệ thống cảnh báo sớm sepsis** cho ICU; các notebook trong thư mục này phục vụ
từng giai đoạn của quá trình đó, không phải là sản phẩm cuối.

## Hiện hành

- **`eda_v2_pipeline_aligned.ipynb`** — EDA hiện dùng, đi theo mạch "phát hiện đặc điểm dữ liệu → chỉ ra đúng lý
  do preprocessing trong `archive/reproduce-xai.ipynb`/`archive/model-comparison.ipynb` (bản tái lập cấu hình
  "Sepsyd best config", H=5, Base+Mask, dùng để chạy TreeSHAP) được thiết kế như vậy". Mỗi mục có phân tích
  "vì sao" ngay dưới biểu đồ của chính mục đó. Chạy được trực tiếp trên toàn bộ dữ liệu thật trong `data/raw/`
  (không lấy mẫu con).

## `reproduction/`

Notebook và tài liệu cho **bước thiết lập model baseline ban đầu** — reproduce lại giải pháp của team Sepsyd
(CinC 2019) để có một model đối chiếu trước khi xây pipeline retraining/serving production. Đây là **một bước**
trong lộ trình xây hệ thống cảnh báo, không phải mục tiêu cuối của dự án.

- `reproduce_sepsis_baseline.ipynb` — bản đang dùng để reproduce baseline theo đúng mô tả trong paper.
- `reproduce_sepsis_best_config.ipynb`, `kaggle_reproduce_sepsis_best_config.ipynb` — biến thể dò cấu hình tốt
  hơn, tham khảo song song với bản baseline.
- `PublishedPaperCinC2019-423.pdf` — paper gốc.

## `archive/`

Notebook lịch sử/tham khảo, giữ lại nguyên nội dung nhưng **không đảm bảo chạy được** sau khi cấu trúc thư mục
đổi (ví dụ `eda_ver1.ipynb`, `data-pipeline.ipynb` nạp `cleaning.py`/`feature-engineering.py` từ vị trí cũ trong
`data/`, nay đã chuyển vào `src/pipelines/processing/`). Dùng để đọc lại tư duy/phân tích cũ, không dùng để chạy
lại trực tiếp.

- `eda_ver1.ipynb` — bản EDA trước `eda_v2_pipeline_aligned.ipynb`.
- `data-pipeline.ipynb` — bản dựng CV-fold thủ công, đã được thay thế bởi `src/pipelines/processing/{silver_pipeline,gold_pipeline}.py`.
- `model-comparison.ipynb`, `reproduce-xai.ipynb` — so sánh model và explainability, thực hiện độc lập.
