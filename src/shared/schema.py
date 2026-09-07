"""PhysioNet Challenge 2019 patient file schema."""

VITAL_SIGNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
]

LABS = [
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]

DEMOGRAPHICS = [
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

REQUIRED_COLUMNS = VITAL_SIGNS + LABS + DEMOGRAPHICS
OPTIONAL_COLUMNS = ["SepsisLabel"]

IMPORTANT_VITAL_COLUMNS = ["HR", "Temp", "SBP", "MAP", "Resp", "O2Sat"]

SUPPORTED_EXTENSIONS = [".psv", ".csv"]
PSV_DELIMITER = "|"
CSV_DELIMITER = ","
