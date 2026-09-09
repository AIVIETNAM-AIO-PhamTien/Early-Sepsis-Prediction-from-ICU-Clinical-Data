# Sample patient data

Sample `.psv` files used for the Streamlit app demo and inference tests.

| File | Set | Description |
|---|---|---|
| `p000001.psv` | A | Non-sepsis patient (~54 ICU hours) |
| `p000003.psv` | A | Patient with `SepsisLabel=1` at some hours |
| `p100001.psv` | B | Patient from hospital system 2 (~24 hours) |

## Schema

Each file: **1 patient**, **1 row = 1 ICU hour**, delimiter `|`.

41 columns: 40 clinical variables + `SepsisLabel`. See details in [`DATASET_OVERVIEW.md`](../../docs/DATASET_OVERVIEW.md).

`patient_id` = filename (not a column), e.g. `p000001.psv` → `p000001`.

## Source

Dataset packaged on Kaggle: [sepsyd-data](https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data) (PhysioNet/CinC Challenge 2019).

The sample files in this repo were downloaded directly from PhysioNet (same origin as the Kaggle dataset):

- Set A: `https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/`
- Set B: `https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/`

To download the **full** dataset via the Kaggle API, see [`scripts/download_kaggle_data.ps1`](../../scripts/download_kaggle_data.ps1).
