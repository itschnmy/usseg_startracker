# Hướng dẫn cài đặt và vận hành

## 1. Phạm vi áp dụng

Quy trình áp dụng cho `prod/version1`, Python 3.10 trên Linux amd64, sử dụng
ảnh BMP/PNG và cấu hình mẫu đã kiểm thử. Hệ thống xử lý từng ảnh theo chế độ
Lost-in-Space (LIS). Các yêu cầu triển khai JPG, HFOV 8,1° và Jetson ARM64
được mô tả trong [đặc tả Jetson](production-jetson.md).

## 2. Cài môi trường — Linux, Python 3.10

```bash
git clone --branch prod/version1 https://github.com/itschnmy/usseg_startracker.git
cd usseg_startracker
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements-runtime.lock
bash examples/materialize_assets.sh
```

Các lệnh được thực hiện tại thư mục gốc của repository.
Script materialize lấy database từ `origin/main` khi thiếu; clone single-branch
cần fetch ref đó trước.

## 3. Kiểm tra hệ thống và xử lý ảnh mẫu

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

Báo cáo tự kiểm tra lưu tại `out/first-selftest/self-test/<ID>/report.json`.
Kết quả đã xác nhận: **8/8 phép kiểm tra đạt**, ảnh03 có **7 sao nhận dạng**,
ảnh Stellarium có **16 sao nhận dạng**. Bộ kiểm tra bao gồm ảnh đen và dữ liệu
hỏng để xác nhận xử lý lỗi. Ảnh Stellarium được lưu sẵn trong repository.
Chi tiết: [báo cáo kiểm thử Podman](podman-recheck-20261105.md).

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

Lệnh `prepare-input` tạo metadata và dấu hiệu ảnh đã sẵn sàng (`.ready`).
Mỗi job có mã định danh riêng và không bị ghi đè. Khi chạy lại cùng cấu hình,
hệ thống bỏ qua kết quả đã hoàn tất. Sử dụng thư mục output mới để đo lại hoặc
khi thay đổi mã nguồn, cấu hình hay database.

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

## 6. Tra cứu kết quả

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

Tài liệu liên quan: [cấu hình xử lý ảnh](image-example-configs.md),
[thu thập ảnh Stellarium](stellarium_image.md).
