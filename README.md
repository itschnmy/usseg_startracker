# USSEG Star Tracker

`prod/version1` — ảnh → phát hiện sao → Tetra → hướng camera. Python **3.10**, chạy local hoặc **Podman**; batch xuất JSON từng ảnh.

## Người mới đọc theo thứ tự

1. [Bắt đầu chạy code](docs/quickstart.md): cài môi trường, chạy ảnh03, Stellarium, batch và Podman; nơi đọc kết quả.
2. [Production Jetson / VinSpace](docs/production-jetson.md): kiến trúc, yêu cầu camera và các bước trước bàn giao.
3. [Runtime chi tiết](docs/version3-runtime.md): schema, watchdog và stop/resume.

**Hiện trạng 05/11/2026:** JPG độ phân giải lớn, resize, database HFOV 8,1° và nghiệm thu ARM64 còn cần hoàn thiện. Preset hiện có là preset lab.

## Chạy nhanh khi đã có môi trường và database

Từ thư mục gốc repo:

```bash
.venv/bin/python -m usseg_runtime.cli self-test \
  --config configs/runtime/development.json --output out/first-selftest
```

Nếu chưa có môi trường, làm theo [quickstart](docs/quickstart.md). Kết quả nằm ở `out/first-selftest/self-test/<ID>/report.json`.

## Kết quả đã kiểm chứng

| Dữ liệu | Kết quả | Giới hạn |
|---|---|---|
| Ảnh03 camera | 7 match; lệch tâm Astrometry khoảng 0,033° | Chưa chấm full attitude bằng WCS gốc |
| Stellarium fixture | 16/16 ID đúng; attitude error 0,002738° | Reference mô phỏng |
| Podman rootless WSL2 amd64 | Self-test 8/8, batch mẫu 4/4 có nghiệm | Chưa nghiệm thu Jetson |

`solved` / `ok` nghĩa là có nghiệm, chưa chứng minh hướng đúng. [Kết quả và bằng chứng](docs/current-results.md).

## Cấu trúc cần biết

| Thư mục | Nội dung |
|---|---|
| `models/` | Detector, Tetra, attitude và pipeline |
| `usseg_runtime/` | Input/output, batch, watchdog, resume |
| `configs/` | Cấu hình ảnh mẫu và runtime |
| `scripts/` | Build/chạy Podman, kiểm tra local |
| `schemas/` | Hợp đồng JSON |
| `dataset/`, `data/` | Fixture và database |
| `docs/` | Hướng dẫn, kiến trúc, bằng chứng |
| `out/` | Output local, không tự commit |

[Mục lục tài liệu](docs/README.md) có hướng dẫn chuyên đề.
