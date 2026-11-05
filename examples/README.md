# Chạy dữ liệu mẫu

Từ thư mục gốc repository, dùng Python 3.10 và database đã chuẩn bị:

```bash
.venv/bin/python -m examples.run_image_examples --output-dir out/image-examples
.venv/bin/python -m examples.run_batch --output out/batch-example
```

`run_image_examples` gọi pipeline theo preset từng ảnh. `run_batch` chuẩn bị
job, chạy runtime và ghi JSON, báo cáo cùng ảnh đánh dấu sao. Thư mục output
của `run_batch` phải là đường dẫn mới.

Quy trình vận hành Podman được mô tả trong [hướng dẫn vận hành](../docs/operation.md).
