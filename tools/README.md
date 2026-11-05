# Công cụ phát triển

Các công cụ này chạy trên máy phát triển và không được đóng gói trong image runtime.
Thực thi module từ thư mục gốc repository bằng Python 3.10.

| Thư mục | Chức năng |
|---|---|
| `stellarium/` | Thu ảnh mô phỏng, metadata và nhãn; xem README trong thư mục |
| `calibration/` | Tạo master dark và xuất ảnh đã hiệu chỉnh |
| `database/` | Tạo database Tetra/K-vector |
| `diagnostics/` | Kiểm tra detector/solver và điều tra ảnh03 |
| `datasets/` | Tải dữ liệu DUST phục vụ đánh giá mặt đất |
| `openmv/` | Mã xử lý ảnh MicroPython dành cho OpenMV/STM32 |

Ví dụ:

```bash
.venv/bin/python -m tools.diagnostics.check_stellarium --help
.venv/bin/python -m tools.database.generate_tetra_database --help
.venv/bin/python -m tools.calibration.build_master_dark --help
```

Mã OpenMV sử dụng API thiết bị, không chạy bằng CPython trên máy tính.
Các công cụ cần ảnh hoặc catalog ngoài repository phải được cấp đường dẫn cụ thể.
