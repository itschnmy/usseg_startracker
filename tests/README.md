# Kiểm thử

Thực thi tại thư mục gốc repository, Python 3.10:

```bash
.venv/bin/python -m pytest -q
bash scripts/verify_cpp.sh out/cpp-validation
```

`python/` kiểm tra thuật toán, convention, cấu hình, runtime và công cụ thu ảnh.
`cpp/` kiểm tra các mô-đun C++ riêng; kết quả không thay cho đánh giá pipeline Python.
Test phụ thuộc CSV ngoài repository được đánh dấu skip khi dữ liệu không có.

Kiểm tra mặt đất đầy đủ:

```bash
bash scripts/verify_local.sh out/local-validation
```

Thư mục output phải chưa tồn tại. Script chạy pytest, runtime self-test,
batch ảnh mẫu và kiểm thử C++ tuần tự. Kiểm tra container dùng
`scripts/run_test.sh` theo [hướng dẫn vận hành](../docs/operation.md).
