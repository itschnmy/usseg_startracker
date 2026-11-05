# Báo cáo kiểm thử — prod/version2

Ngày thực hiện: **05/11/2026**. Môi trường Linux WSL2 amd64, Podman 4.9.3,
Python 3.10. Thuật toán và preset được giữ nguyên khi tổ chức lại mã nguồn.

## Kết quả version2

| Hạng mục | Kết quả |
|---|---|
| Python | 105 passed, 2 skipped do thiếu CSV bên ngoài |
| C++ | LIS, tracking và preprocessing đạt |
| Batch local | 4/4 ảnh có nghiệm |
| Podman self-test | 8/8 đạt, gồm kiểm tra resume và xử lý input lỗi |
| Batch Podman | 4/4 ảnh có nghiệm, exit 0 |
| Đối chiếu version1 | Quaternion, RA/Dec và ID sao giống hệt trên bốn ảnh mẫu |
| Nội dung image | Không chứa examples, tools hoặc tests |
| OCI archive | Đã export, kiểm tra SHA256 đạt |

Revision mã nguồn runtime: `bf3d2ad7dc27dbbfc7bd6fb55b4b580d983c76fb`.
Image `localhost/usseg:prod-version2`:
`3e29c8d47e84599a8b68e547cc3419b3d28cad743ef23a6407ea21468ce1fb2b`.

| Ảnh | Sao nhận dạng | Worker time (ms) |
|---|---:|---:|
| 03.bmp | 7 | 108.405 |
| 200701 raw | 7 | 27.018 |
| 200701 calibrated | 7 | 101.745 |
| Stellarium | 16 | 274.690 |

Timing là một lượt smoke trên bốn ảnh, không dùng để kết luận cải thiện tốc độ.
Image sử dụng lại layer dependency đã kiểm chứng, cùng lockfile; chưa hoàn tất
phép build từ base sạch qua mạng trong lượt này. Chưa nghiệm thu ARM64,
JPG lớn hoặc giới hạn 4 GB/4 core.

[Kết quả máy đọc](evidence/prod-version2-validation.json).
Raw output: `out/prod-version2-podman/`, `out/prod-version2-validation-local/`,
`out/prod-version2-cpp/`. Bản export: `out/prod-version2-release/`.

## Đối chứng version1 — 05/11/2026

## Phạm vi kiểm tra

Podman rootless 4.9.3, WSL2 amd64, Python 3.10; image
`localhost/usseg:prod-version1`, ID
`00a911d7dfae216e877aa2f0a028a7c074975a7a63dd7be919729d20d19c6d2c`.
Source revision `d494f0f97ea783ed9baec19520ba12bfd4677923`.

Chạy lại image revision trên. SHA256 của **38 file Python/JSON** trong
models/utils/usseg_runtime/configs khớp source local tại thời điểm kiểm tra.
Network none, root/input/config chỉ đọc, output mount ghi. Chưa áp giới hạn
4 GB/4 core và chưa chạy Jetson.

## Kết quả đo mới

Self-test **8 passed, 0 failed**, gồm ảnh đen, payload lỗi, unit quaternion,
reference Stellarium và resume không nhân đôi kết quả. Batch **4/4 solved**, exit 0.

| Ảnh | Match | RA / Dec (độ) | Worker time (ms) |
|---|---:|---|---:|
| 03.bmp | 7 | 48.816336 / 41.369726 | 193.711 |
| 200701 raw | 7 | 267.250966 / −31.683422 | 54.752 |
| 200701 calibrated | 7 | 267.256413 / −31.683188 | 35.885 |
| Stellarium | 16 | 291.950409 / 41.128080 | 427.991 |

Pose và catalog IDs của cả bốn ảnh giống hệt baseline Podman 04/11/2026.
Chấm lại Stellarium với labels: **16/16 cặp đúng**, attitude error
**0.002737686386468914°**, boresight error **0.0022887066360223777°**.
Ảnh03 chỉ có đối chiếu tâm khoảng 0,033° từ screenshot Astrometry;
chưa xác nhận full attitude bằng WCS gốc.

Worker P50 **124.232 ms**, P95 **392.849 ms** trên bốn ảnh khác nhau.
Database-load **1.9996 s**; peak worker RSS **223352 KiB**, khoảng **218,1 MiB**,
không phải tổng RAM container. `compute_fps=5.6153` tính bằng 4 ảnh / tổng
worker time, không phải FPS camera hoặc throughput toàn batch.
Timing mới khác lần trước là bình thường; bốn ảnh chưa đủ kết luận tăng/giảm tốc độ.

## Tái hiện và bằng chứng

```bash
USSEG_IMAGE=localhost/usseg:prod-version1 bash scripts/run_test.sh \
  out/prod-version1-validation/input out/prod-docs-recheck-20261105 configs/runtime
```

Đây là lệnh đã chạy trên workspace có sẵn bốn job. Người mới dùng
[quickstart](operation.md) để tạo input; chọn output mới để đo lại.

Raw evidence local, không tự commit:

- `out/prod-docs-recheck-20261105/self-test/ddc785b4afae4ca1a6f22c6345d8a0cb/report.json`.
- `out/prod-docs-recheck-20261105/runs/20261105T150526Z-4537b95acd63/summary.json`.
- Cùng run có `results/*.json`, `summary.csv`, `run.json`, `events.jsonl`.
- `out/prod-docs-recheck-20261105/container-source-hashes.json`.

Run ID dùng UTC; lượt chạy diễn ra tối 05/11/2026 tại Việt Nam.

## Phạm vi hỗ trợ

Source vẫn xác nhận decoder runtime từ chối JPEG, max_pixels 16.777.216,
database 10–30°, chưa resize và chưa nghiệm thu ARM64/FOV 8,1°.
Các lệnh self-test và CLI ảnh03/Stellarium trong quickstart đã chạy thành công.
Không diễn giải các test này thành bằng chứng hỗ trợ camera bay mới.
