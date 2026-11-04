# Tài liệu USSEG

## Tài liệu chính của version3

| Tài liệu | Nội dung |
|---|---|
| [Kế hoạch version3](version3-plan.md) | Podman, camera RAW, input/output, lifecycle, test matrix và acceptance gates |
| [Hướng dẫn runtime](version3-runtime.md) | Lệnh đã triển khai, schema draft, stop/resume, build/run Podman |
| [Báo cáo triển khai](version3-report.md) | Kết quả kiểm thử thực tế, camera smoke, các phần chờ hardware |
| [Hiện trạng và kết quả](current-results.md) | Pipeline đang chạy, ảnh03/200701/Stellarium, LOST-evals/DUST và giới hạn |

Version3 đã có runtime và Containerfile; chưa kiểm chứng Podman trên máy đích.
Bên phần cứng đọc hướng dẫn runtime và các điều kiện nghiệm thu trong kế hoạch.

## Hướng dẫn còn dùng

- [Fixtures và ảnh kết quả](../dataset/README.md).
- [Dark-frame8-bit](dark-frame.md).
- [Stellarium capture](stellarium_image.md).
- [Ubuntu setup](ubuntu.md).
- [Preprocessing/presets](image-example-configs.md).
- [Đối chứng ảnh03](image03-experimental.md).
- [Audit version01](version01-validation.md): lịch sử, không phải trạng thái v3.

Docs01–07 cũ được loại khỏi version3; Git commit `33d9c17` giữ nguyên nội dung.
Các báo cáo cũ trong `out/` là evidence lịch sử; không đọc một kết luận no_solve
cũ thành kết quả của preset mới.
