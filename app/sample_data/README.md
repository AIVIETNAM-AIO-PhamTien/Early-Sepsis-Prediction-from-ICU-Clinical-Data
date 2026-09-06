# Sample patient data

Các file `.psv` mẫu phục vụ demo app Streamlit và test inference.

| File | Set | Mô tả |
|---|---|---|
| `p000001.psv` | A | Bệnh nhân không sepsis (~54 giờ ICU) |
| `p000003.psv` | A | Bệnh nhân có `SepsisLabel=1` ở một số giờ |
| `p100001.psv` | B | Bệnh nhân từ hospital system 2 (~24 giờ) |

## Schema

Mỗi file: **1 bệnh nhân**, **1 dòng = 1 giờ ICU**, delimiter `|`.

41 cột: 40 biến lâm sàng + `SepsisLabel`. Xem chi tiết trong [`DATASET_OVERVIEW.md`](../../DATASET_OVERVIEW.md).

`patient_id` = tên file (không có trong cột), ví dụ `p000001.psv` → `p000001`.

## Nguồn

Dataset đóng gói trên Kaggle: [sepsyd-data](https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data) (PhysioNet/CinC Challenge 2019).

Các file mẫu trong repo được tải trực tiếp từ PhysioNet (cùng nguồn gốc với Kaggle dataset):

- Set A: `https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/`
- Set B: `https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/`

Để tải **toàn bộ** dataset qua Kaggle API, xem [`scripts/download_kaggle_data.ps1`](../../scripts/download_kaggle_data.ps1).
