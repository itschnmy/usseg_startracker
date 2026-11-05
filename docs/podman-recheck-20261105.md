# Kiểm chứng docs và Podman — 05/11/2026

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
[quickstart](quickstart.md) để tạo input; chọn output mới để đo lại.

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
