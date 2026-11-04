# USSEG Star Tracker

Nhánh `dev/version3` chuẩn bị đóng gói pipeline ảnh camera → attitude bằng
**Podman**, Python **3.10**. Base code: `dev/version2` tại `33d9c17`.

**Hiện tại:** đã triển khai runtime batch/watch/self-test, schema draft1.0,
watchdog và persistent output; có Containerfile và scripts Podman.
Chưa build/run OCI trên máy đích; camera profiles kèm repo là cấu hình lab.

- [Kế hoạch version3 và hợp đồng input/output](docs/version3-plan.md)
- [Hướng dẫn runtime và Podman](docs/version3-runtime.md)
- [Báo cáo triển khai version3](docs/version3-report.md)
- [Báo cáo kết quả, pipeline Mermaid và ảnh minh họa](docs/current-results.md)
- [Mục lục tài liệu](docs/README.md)

## Chạy code hiện có

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash examples/materialize_assets.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m examples.run_image_examples
python -m pytest examples/tests -q
```

Script materialize cần asset trong Git refs; thực hiện khi chuẩn bị môi trường.
Bản container tương lai sẽ mang sẵn database và dependencies, không tải lúc chạy.

Chạy kiểm thử mặt đất tuần tự (Python, self-test, camera smoke, C++ nếu đủ toolchain),
lưu cùng một folder mới; các target thiếu dependency được ghi rõ skip:

```bash
bash scripts/verify_local.sh out/v3-verification-new
```

Để kiểm từng ảnh camera bằng runtime, xem `prepare-input`, `batch`, `watch` trong
[hướng dẫn version3](docs/version3-runtime.md). Bên vận hành có Podman dùng
`scripts/run_test.sh INPUT_DIR OUTPUT_DIR CONFIG_DIR` để chạy self-test rồi batch.

Chạy một ảnh:

```bash
.venv/bin/python -m models.pipeline.cli \
  --config configs/examples/image03-experimental.json \
  --image dataset/03.bmp --output out/image03.json
```

`status=ok` nghĩa là solver tìm được nghiệm. Độ chính xác attitude cần reference
độc lập. CLI exit0 cũng có thể là `no_solve`; luôn đọc status trong JSON.

## Kết quả làm baseline

| Dữ liệu | Kết quả đã đo | Giới hạn |
|---|---|---|
| Ảnh03 camera / preset experimental | 7 matches, tâm lệch khoảng0.033° so Astrometry screenshot | Chưa có WCS gốc để chấm full attitude |
| Ảnh200701 raw và calibrated | Cùng7 matches | Chưa có WCS/reference riêng; dùng FOV lab16.8° |
| Stellarium fixture | 16/16 ID pairs đúng, attitude error0.002738° | Reference từ simulator |
| Python tests version3, 2026-11-04 | 105 passed, 2 skipped | RAW camera bay chưa được nghiệm thu |
| Podman4.9.3 rootless / WSL2 amd64 | Self-test8/8, camera4/4 solved; OCI load và resume pass | [Evidence](docs/evidence/version3/podman-validation.json); chưa nghiệm thu host bay |

[Báo cáo chi tiết](docs/current-results.md) chứa số liệu LOST-evals/DUST lịch sử,
source evidence và các hạn chế. Chưa có bằng chứng flight readiness.

## Camera và vận hành

Single-image API hỗ trợ encoded/grayscale và optional dark8-bit. Runtime mới
thêm native-depth calibration và unpacked adapter; RG8/RG10/RG12 cần được đối
chiếu pixel format, packing/stride/endianness với mẫu SDK thật của camera.
Các mode sensor/ROI/binning phải có profile intrinsics/calibration tương ứng.

V3 dùng `/input`, `/config`, `/output`, có ready marker, watch/resume và watchdog.
Decoder có Mono/Bayer RG8/10/12 unpacked explicit layout; profiles mặc định chỉ
bật encoded8, packed RAW chờ SDK samples. Hợp đồng với hardware vẫn là draft.

## Tài liệu thực hành

- [Ảnh fixture và nguồn phần cứng](dataset/README.md).
- [Dark-frame calibration hiện tại](docs/dark-frame.md).
- [Thu ảnh Stellarium](docs/stellarium_image.md), [setup Ubuntu](docs/ubuntu.md).
- [Preset và preprocessing](docs/image-example-configs.md).
- [Thử nghiệm verifier ảnh03](docs/image03-experimental.md).
- [Audit version01 — lịch sử](docs/version01-validation.md).

Repo giữ Python pipeline tại `models/`, scripts/tests tại `examples/`,
camera capture tại `tools/stellarium/`, C++ thử nghiệm tại `models/cpp/`.
Kết quả chạy mới lưu dưới `out/` và không tự commit. Các artifact chọn làm bằng
chứng báo cáo nằm trong `docs/evidence/` và `docs/assets/`.

Các docs đánh số01–07 đã được thay bằng bộ tài liệu trên; có thể truy lại bản cũ
trong commit `33d9c17`. Chi tiết dọn tài liệu ở cuối báo cáo hiện trạng.
