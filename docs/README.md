# Tài liệu USSEG

## Lộ trình cho người mới

| Thứ tự | Tài liệu | Mục đích |
|---|---|---|
| 1 | [Bắt đầu chạy code](quickstart.md) | Cài môi trường, chạy mẫu, batch/Podman và đọc JSON |
| 2 | [Production Jetson / VinSpace](production-jetson.md) | Kiến trúc, input/output, phần còn thiếu và test bàn giao |
| 3 | [Runtime chi tiết](version3-runtime.md) | Schema, watchdog, stop/resume |

## Chỉ đọc khi cần

- [Kiểm chứng Podman 05/11/2026](podman-recheck-20261105.md): kết quả mới và audit docs.

- [Kết quả và bằng chứng](current-results.md): ảnh mẫu và benchmark lịch sử.
- [Nguồn ảnh mẫu](../dataset/README.md).
- [Preset ảnh](image-example-configs.md) và [đối chứng ảnh03](image03-experimental.md).
- [Dark frame](dark-frame.md).
- [Tạo ảnh Stellarium](stellarium_image.md).
- [Thiết lập Ubuntu mở rộng](ubuntu.md).

## Tài liệu cũ đã dọn

Đã bỏ `version3-plan.md`, `version3-report.md`, `version01-validation.md` và `summary_old.html` khỏi cây tài liệu hiện hành. Kế hoạch mới ở `production-jetson.md`; evidence và ảnh minh họa được giữ lại.

Có thể khôi phục từ commit `d494f0f`, ví dụ:

```bash
git show d494f0f:docs/version3-report.md
```

Số liệu lịch sử trong `current-results.md` không phải cam kết hiệu năng Jetson/camera mới.
