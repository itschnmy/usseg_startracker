# USSEG Star Tracker

Phần mềm xác định hướng camera từ ảnh sao, gồm phát hiện tâm sao, nhận dạng bằng Tetra và ước lượng tư thế bằng SVD. Phiên bản `prod/version2` sử dụng Python 3.10, triển khai bằng Podman và xuất kết quả JSON theo từng ảnh.

## Tài liệu triển khai

1. [Hướng dẫn cài đặt và vận hành](docs/operation.md).
2. [Đặc tả triển khai Jetson / VinSpace](docs/deployment.md).
3. [Báo cáo kiểm thử](docs/validation.md).

**Hiện trạng 05/11/2026:** JPG độ phân giải lớn, resize, database HFOV 8,1° và nghiệm thu ARM64 còn cần hoàn thiện. Preset hiện có là preset lab.

## Kiểm tra hệ thống

Từ thư mục gốc repo:

```bash
.venv/bin/python -m usseg_runtime.cli self-test \
  --config configs/runtime/development.json --output out/first-selftest
```

Quy trình cài đặt được mô tả trong [hướng dẫn vận hành](docs/operation.md). Kết quả kiểm tra lưu tại `out/first-selftest/self-test/<ID>/report.json`.

## Kết quả đã kiểm chứng

| Dữ liệu | Kết quả | Giới hạn |
|---|---|---|
| Ảnh03 camera | 7 match; lệch tâm Astrometry khoảng 0,033° | Chưa chấm full attitude bằng WCS gốc |
| Stellarium fixture | 16/16 ID đúng; attitude error 0,002738° | Reference mô phỏng |
| Podman rootless WSL2 amd64 | Self-test 8/8, batch mẫu 4/4 có nghiệm | Chưa nghiệm thu Jetson |

Trạng thái `solved` / `ok` cho biết thuật toán đã trả nghiệm. Độ chính xác được đánh giá riêng bằng dữ liệu tham chiếu. [Báo cáo kết quả](tools/diagnostics/baseline.md).

## Cấu trúc mã nguồn

| Thư mục | Nội dung |
|---|---|
| `models/` | Detector, Tetra, attitude và pipeline |
| `usseg_runtime/` | Input/output, batch, watchdog, resume |
| `configs/` | Cấu hình ảnh mẫu và runtime |
| `scripts/` | Build/chạy Podman, kiểm tra local |
| `schemas/` | Hợp đồng JSON |
| `dataset/`, `data/` | Fixture và database |
| `docs/` | Triển khai, vận hành và kiểm thử |
| `tests/` | Kiểm thử Python và C++ |
| `tools/` | Công cụ phát triển và thu thập dữ liệu |
| `examples/` | Chạy ảnh mẫu và batch mẫu |
| `out/` | Output local, không tự commit |

Công cụ phát triển nằm trong `tools/`; kiểm thử Python/C++ nằm trong `tests/`.
Container chỉ đóng gói runtime, thuật toán, cấu hình và dữ liệu tự kiểm tra.
