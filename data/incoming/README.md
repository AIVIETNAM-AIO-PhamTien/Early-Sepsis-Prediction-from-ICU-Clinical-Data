# Incoming retraining batches

Đặt batch dữ liệu mới cần retrain vào thư mục này. Hỗ trợ:

- Các file bệnh nhân `.psv`.
- File `.zip` chứa `.psv`.
- Canonical `.csv` có `patient_id`, đủ 40 biến clinical và `SepsisLabel`.

Không đặt fixed-test patients vào đây. DAG sẽ bỏ qua batch có checksum đã xử lý. Mọi `patient_id` đúng dạng `pXXXXXX` đều thuộc cùng một cohort, không phân Set A/B.
