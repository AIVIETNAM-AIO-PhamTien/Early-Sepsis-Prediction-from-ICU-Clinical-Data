# Model bootstrap directory

Sau lần chạy DAG retraining thành công, model production được lưu trong một thư mục version, ví dụ:

```text
artifacts/
├── current_model.json
└── models/
    └── model_v1/
        ├── model.json
        ├── preprocessor.json
        ├── feature_config.json
        ├── feature_schema.json
        ├── threshold.json
        ├── metrics.json
        ├── cv_results.json
        ├── utility_config.json
        └── metadata.json
```

`current_model.json` mẫu:

```json
{
  "model_version": "model_v1",
  "dataset_version": "dataset_v1",
  "model_path": "artifacts/models/model_v1/model.json",
  "preprocessor_path": "artifacts/models/model_v1/preprocessor.json",
  "feature_config_path": "artifacts/models/model_v1/feature_config.json",
  "pipeline": "team_v1",
  "model_format": "xgboost_json",
  "threshold": 0.45,
  "status": "current"
}
```

DAG luôn promote candidate mới sau khi candidate vượt qua data-quality checks, train và evaluation thành công. DAG không so sánh candidate với model cũ; `current_model.json` được tạo hoặc ghi đè tự động khi promote.
