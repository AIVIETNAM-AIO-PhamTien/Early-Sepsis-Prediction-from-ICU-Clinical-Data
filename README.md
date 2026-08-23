# AIO-Microwave-Sepsis

Reproduce baseline từ paper "Automated Prediction of Sepsis Onset Using Gradient Boosted Decision Trees" (team Sepsyd, CinC 2019) — dự đoán sớm sepsis từ dữ liệu lâm sàng ICU, PhysioNet/CinC Challenge 2019.

## Hiện trạng project

- **`original/`** — mã nguồn **inference gốc của chính tác giả** (team Sepsyd), tải từ repo chính thức `physionetchallenges/2019ChallengeEntries`. Gồm `get_sepsis_score.py` (thuật toán inference), model XGBoost đã train sẵn (`f120d4e02n8010val434.pickle.dat`), `driver.py`, paper gốc kèm theo. Lưu ý: đây **chỉ có code inference, không có code training** — tác giả không công khai phần này.
- **`docs/reproduce_sepsis_baseline.ipynb`** — nơi đang thực hiện việc **reproduce lại baseline** theo mô tả thuật toán trong paper (preprocessing, feature engineering, train XGBoost, đánh giá bằng utility score chính thức của Challenge).
- **`docs/PublishedPaperCinC2019-423.pdf`** — bản paper gốc.
- **`DATASET_OVERVIEW.md`** — tài liệu mô tả dataset PhysioNet Challenge 2019.
- **`requirements.txt`** — dependency cho phần reproduce (numpy, pandas, scikit-learn, xgboost).

## Trạng thái hiện tại

Chưa có dữ liệu training trong project — đang chờ quyết định cách lấy data (chưa chạy được `reproduce_sepsis_baseline.ipynb` cho đến khi có data).
