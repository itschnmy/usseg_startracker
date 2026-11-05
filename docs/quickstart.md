# Bắt đầu chạy USSEG

## 1. Đọc theo thứ tự

1. Trang này: chạy thử và đọc kết quả.
2. [Production Jetson](production-jetson.md): yêu cầu VinSpace và phần còn thiếu.
3. [Runtime chi tiết](version3-runtime.md): schema, timeout, stop/resume.

Các lệnh dưới dùng ảnh mẫu BMP/PNG đã kiểm chứng trên amd64. JPG lớn, resize,
FOV 8,1° và Jetson ARM64 chưa được nghiệm thu. Pipeline chạy LIS từng ảnh.

## 2. Cài môi trường — Linux, Python 3.10

```bash
git clone --branch prod/version1 https://github.com/itschnmy/usseg_startracker.git
cd usseg_startracker
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements-runtime.lock
bash examples/materialize_assets.sh
```

Nếu đã clone, dùng repo hiện có. Luôn chạy từ thư mục gốc repo.
Script materialize lấy database từ `origin/main` khi thiếu; clone single-branch
cần fetch ref đó trước.

## 3. Chạy tự kiểm tra và hai ảnh mẫu

```bash
python -m usseg_runtime.cli self-test \
  --config configs/runtime/development.json --output out/first-selftest

python -m models.pipeline.cli \
  --config configs/examples/image03-experimental.json \
  --image dataset/03.bmp --output out/image03.json

python -m models.pipeline.cli \
  --config configs/examples/stellarium.json \
  --image dataset/stellarium_test/stellarium_view.png --output out/stellarium.json
```

Self-test: xem `out/first-selftest/self-test/<ID>/report.json`; baseline **8/8 pass**.
Có ảnh đen/payload lỗi cố ý, nên không yêu cầu mọi ảnh đều solved.
Ảnh03 có **7 match**, Stellarium **16 match**. Không cần mở Stellarium.
[Đo lại Podman 05/11/2026](podman-recheck-20261105.md) có số liệu và giới hạn.

## 4. Chuẩn bị input và chạy batch

```bash
python -m usseg_runtime.cli prepare-input \
  --image dataset/03.bmp --input out/demo-input --job-id image03 --profile lab640
python -m usseg_runtime.cli prepare-input \
  --image dataset/stellarium_test/stellarium_view.png --input out/demo-input \
  --job-id stellarium --profile stellarium1920
python -m usseg_runtime.cli batch \
  --input out/demo-input --output out/demo-output --config configs/runtime/development.json
```

Helper tạo metadata và ready marker; chỉ copy ảnh vào folder chưa đủ.
Job đã tồn tại không bị ghi đè. Chạy lại batch cùng deployment/output sẽ resume;
muốn đo lại, dùng output mới. Đổi source/config/database cũng cần output mới.

Kết quả: `out/demo-output/runs/<run_id>/summary.json`, `summary.csv` và
`results/*.json`. CSV ánh xạ job với file kết quả.

## 5. Chạy qua Podman

Build trên host có Podman, source đã commit và working tree sạch:

```bash
USSEG_IMAGE=localhost/usseg:prod-version1 USSEG_PYTHON=.venv/bin/python \
  bash scripts/build_podman.sh out/prod-release
USSEG_IMAGE=localhost/usseg:prod-version1 \
  bash scripts/run_test.sh out/demo-input out/demo-podman-output configs/runtime
```

Script chạy self-test rồi batch. Dùng input đã tạo ở bước 4 và output riêng cho
container. `configs/runtime/runtime.json` dùng đường dẫn container;
`development.json` dùng đường dẫn local.

Nếu nhận archive thay vì tự build:

```bash
cd out/prod-release
sha256sum -c SHA256SUMS
podman load -i usseg-v3.oci.tar
cd ../..
```

Archive phải đúng kiến trúc host. Build hiện theo kiến trúc máy build;
archive amd64 không chạy native trên Jetson ARM64. Tên archive vẫn là
`usseg-v3.oci.tar` theo script hiện tại.

## 6. Đọc JSON

| Nội dung | CLI một ảnh | Runtime / Podman |
|---|---|---|
| Có nghiệm | `status=ok` | `status=solved` |
| Hướng nhìn | `ra_deg`, `dec_deg` | `pose.ra_deg`, `pose.dec_deg` |
| Quaternion | `quaternion_inertial_to_body_wxyz` | `pose.quaternion_wxyz` |
| Số sao match | `num_matches` | `algorithm.num_matches` |
| Thời gian | `timings` | `timings_ns`, `worker_total_ns` |
| Không giải được | `no_solve` | `no_solve`, `pose=null` |

Quaternion là catalog→camera, wxyz; tên legacy “body” ở CLI chưa có mounting
transform. Có nghiệm chưa chứng minh đúng: accuracy cần reference độc lập.
Timing ns chia 1.000.000 ra ms; không coi compute FPS là tốc độ chụp camera.

Muốn chỉnh preset: [cấu hình ảnh](image-example-configs.md).
Muốn tự tạo dữ liệu: [Stellarium](stellarium_image.md).
