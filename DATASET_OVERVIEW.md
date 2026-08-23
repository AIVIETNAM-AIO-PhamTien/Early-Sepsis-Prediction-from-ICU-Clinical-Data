# Dataset Documentation — PhysioNet Challenge 2019

## Tổng quan Dataset

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên** | PhysioNet/Computing in Cardiology Challenge 2019 |
| **Nhiệm vụ** | Dự báo sớm Sepsis từ dữ liệu lâm sàng ICU (trước 6 giờ) |
| **Nguồn chính** | [PhysioNet Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/) |
| **DOI** | https://doi.org/10.13026/v64v-d857 |
| **Ngày phát hành** | 5/8/2019 |
| **License** | Open Access |

---

## Bối cảnh và Mục tiêu

**Sepsis** là tình trạng nguy hiểm đến tính mạng xảy ra khi phản ứng của cơ thể với nhiễm trùng gây tổn thương mô, suy tạng hoặc tử vong.

- Tại Mỹ: ~1.7 triệu người mắc và ~270,000 tử vong mỗi năm
- Toàn cầu: ~30 triệu người mắc và ~6 triệu tử vong mỗi năm
- Chi phí điều trị tại Mỹ: $24 tỷ/năm (13% tổng chi phí y tế)

**Mục tiêu:** Xây dựng thuật toán tự động phát hiện nguy cơ Sepsis **trước 6 giờ** so với thời điểm chẩn đoán lâm sàng chính thức.

---

## Định nghĩa Sepsis (Sepsis-3)

Sepsis được xác định dựa trên hai điều kiện cùng xảy ra:

### 1. t_suspicion — Nghi ngờ nhiễm trùng
- Timestamp sớm nhất của cặp: kháng sinh IV + cấy máu
- Nếu kháng sinh được cho trước → cấy máu phải được lấy trong **24 giờ**
- Nếu cấy máu trước → kháng sinh phải được chỉ định trong **72 giờ**
- Kháng sinh phải được dùng liên tục ít nhất **72 giờ**

### 2. t_SOFA — Suy tạng
- Sự suy giảm **≥ 2 điểm SOFA** (Sequential Organ Failure Assessment) trong vòng 24 giờ

### 3. t_sepsis — Thời điểm khởi phát
```
t_sepsis = min(t_suspicion, t_SOFA)
```
Điều kiện: `t_suspicion - 24 <= t_SOFA <= t_suspicion + 12`

---

## Cấu trúc Dữ liệu

### Format file
- Mỗi bệnh nhân = 1 file `.psv` (pipe-separated values)
- Mỗi **hàng** = dữ liệu đo lường của **1 giờ** trong ICU
- Mỗi **cột** = 1 biến đo lường theo thời gian
- Giá trị `NaN` = không có phép đo tại thời điểm đó

### Ví dụ format file
```
HR|O2Sat|Temp|...|HospAdmTime|ICULOS|SepsisLabel
NaN|  NaN| NaN|...|        -50|     1|          0
 86|   98| NaN|...|        -50|     2|          0
 75|  NaN| NaN|...|        -50|     3|          1
 99|  100|35.5|...|        -50|     4|          1
```

---

## Danh sách 41 Features

### Vital Signs (Cột 1–8)

| Cột | Tên | Mô tả | Đơn vị |
|-----|-----|-------|--------|
| 1 | HR | Heart rate — Nhịp tim | beats/min |
| 2 | O2Sat | Pulse oximetry — SpO2 | % |
| 3 | Temp | Temperature — Nhiệt độ | °C |
| 4 | SBP | Systolic BP — Huyết áp tâm thu | mm Hg |
| 5 | MAP | Mean arterial pressure — Áp lực động mạch trung bình | mm Hg |
| 6 | DBP | Diastolic BP — Huyết áp tâm trương | mm Hg |
| 7 | Resp | Respiration rate — Nhịp thở | breaths/min |
| 8 | EtCO2 | End tidal CO2 — CO2 cuối thì thở ra | mm Hg |

### Laboratory Values (Cột 9–34)

| Cột | Tên | Mô tả | Đơn vị |
|-----|-----|-------|--------|
| 9 | BaseExcess | Excess bicarbonate — Lượng bicarbonate dư | mmol/L |
| 10 | HCO3 | Bicarbonate | mmol/L |
| 11 | FiO2 | Fraction of inspired O2 — Phân suất oxy hít vào | % |
| 12 | pH | pH máu | — |
| 13 | PaCO2 | Partial pressure CO2 (arterial) | mm Hg |
| 14 | SaO2 | O2 saturation (arterial) — Độ bão hòa oxy máu động mạch | % |
| 15 | AST | Aspartate transaminase | IU/L |
| 16 | BUN | Blood urea nitrogen — Nitơ urê máu | mg/dL |
| 17 | Alkalinephos | Alkaline phosphatase | IU/L |
| 18 | Calcium | Canxi máu | mg/dL |
| 19 | Chloride | Clo máu | mmol/L |
| 20 | Creatinine | Creatinine — Chỉ số chức năng thận | mg/dL |
| 21 | Bilirubin_direct | Bilirubin trực tiếp | mg/dL |
| 22 | Glucose | Glucose huyết thanh | mg/dL |
| 23 | Lactate | Lactic acid — Axit lactic | mg/dL |
| 24 | Magnesium | Magie máu | mmol/dL |
| 25 | Phosphate | Phosphate máu | mg/dL |
| 26 | Potassium | Kali máu | mmol/L |
| 27 | Bilirubin_total | Bilirubin toàn phần | mg/dL |
| 28 | TroponinI | Troponin I — Chỉ số tổn thương tim | ng/mL |
| 29 | Hct | Hematocrit — Tỷ lệ hồng cầu | % |
| 30 | Hgb | Hemoglobin | g/dL |
| 31 | PTT | Partial thromboplastin time — Thời gian đông máu | seconds |
| 32 | WBC | Leukocyte count — Bạch cầu | count×10³/µL |
| 33 | Fibrinogen | Fibrinogen — Yếu tố đông máu | mg/dL |
| 34 | Platelets | Tiểu cầu | count×10³/µL |

### Demographics (Cột 35–40)

| Cột | Tên | Mô tả | Đơn vị |
|-----|-----|-------|--------|
| 35 | Age | Tuổi (bệnh nhân >=90 tuổi ghi là 100) | years |
| 36 | Gender | Giới tính: 0 = Nữ, 1 = Nam | — |
| 37 | Unit1 | ICU type: MICU (Medical ICU) | binary |
| 38 | Unit2 | ICU type: SICU (Surgical ICU) | binary |
| 39 | HospAdmTime | Giờ giữa nhập viện và nhập ICU (thường âm) | hours |
| 40 | ICULOS | ICU length-of-stay — Số giờ kể từ khi nhập ICU | hours |

### Target Variable (Cột 41)

| Cột | Tên | Mô tả |
|-----|-----|-------|
| 41 | **SepsisLabel** | `1` nếu `t >= t_sepsis - 6` (6 giờ trước onset); `0` ngược lại |

> **Lưu ý quan trọng:** Label đã được dịch **sớm hơn 6 giờ** so với thời điểm khởi phát lâm sàng thực tế. Điều này nghĩa là model phải **dự đoán Sepsis trước khi nó xảy ra 6 giờ**.

---

## Thống kê Dataset

| Thông tin | Training Set A | Training Set B | Tổng |
|-----------|---------------|----------------|------|
| **Số bệnh nhân** | 20,336 | 20,000 | **40,336** |
| **Tiền tố file** | `p000001.psv` | `p100001.psv` | — |
| **Nguồn bệnh viện** | Hospital system 1 | Hospital system 2 | 2 hệ thống |
| **Dạng file** | `.psv` (pipe-separated) | `.psv` (pipe-separated) | — |

> **Lưu ý:** Còn 1 tập test ẩn từ hệ thống bệnh viện thứ 3 (không công khai), dùng để chấm điểm cuộc thi.

---

## Hàm Đánh giá (Utility Score)

Challenge sử dụng hàm utility tùy chỉnh thay vì AUC hay accuracy thông thường:

| Trường hợp | Điểm |
|-----------|------|
| Dự đoán **đúng**, sớm 12–3 giờ trước t_sepsis | +1.0 (tối đa) |
| Dự đoán **rất sớm** (>12 giờ trước) | -0.05 |
| Dự đoán **muộn** (sau t_sepsis) | -2.0 (tối đa) |
| False Positive trên bệnh nhân không có Sepsis | -0.05 |
| Không dự đoán trên bệnh nhân không có Sepsis | 0 |

**Normalized Score:**
```
U_normalized = (U_total - U_no_predictions) / (U_optimal - U_no_predictions)
```
- Score tốt nhất = 1.0
- Không dự đoán gì = 0.0

---

## Cấu trúc thư mục Local

Chạy `python scripts/setup_data.py` từ thư mục gốc project để tự tải dữ liệu (xem `docs/reproduce_sepsis_baseline.ipynb`). Cấu trúc sau khi tải xong:

```
data/
├── raw/
│   ├── training_setA/
│   │   ├── p000001.psv                      # 20,336 files
│   │   ├── p000002.psv
│   │   └── ...
│   └── training_setB/
│       ├── p100001.psv                      # 20,000 files
│       ├── p100002.psv
│       └── ...
└── processed/                               # Dữ liệu đã xử lý (cache của notebook)

external/
└── evaluate_sepsis_score.py                 # Script chấm utility score chính thức (physionetchallenges/evaluation-2019)
```

> Không upload dữ liệu thô lên GitHub.

---

## Citation

```bibtex
@article{PhysioNet-challenge-2019-1.0.0,
  author = {Reyna, Matthew and Josef, Chris and Jeter, Russell and
            Shashikumar, Supreeth and Moody, Benjamin and
            Westover, M. Brandon and Sharma, Ashish and
            Nemati, Shamim and Clifford, Gari D.},
  title  = {{Early Prediction of Sepsis from Clinical Data:
             The PhysioNet/Computing in Cardiology Challenge 2019}},
  journal = {{PhysioNet}},
  year   = {2019},
  month  = aug,
  note   = {Version 1.0.0},
  doi    = {10.13026/v64v-d857},
  url    = {https://doi.org/10.13026/v64v-d857}
}
```

**Paper chính:**
> Reyna MA, et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge.* Critical Care Medicine 48(2):210-217 (2019). https://doi.org/10.1097/CCM.0000000000004145