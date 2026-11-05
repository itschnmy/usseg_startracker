# Hướng dẫn cài đặt và vận hành

## 1. Phạm vi áp dụng

Quy trình áp dụng cho `prod/version2`, Python 3.10 trên Linux amd64, sử dụng
ảnh BMP/PNG và cấu hình mẫu đã kiểm thử. Hệ thống xử lý từng ảnh theo chế độ
Lost-in-Space (LIS). Các yêu cầu triển khai JPG, HFOV 8,1° và Jetson ARM64
được mô tả trong [đặc tả Jetson](deployment.md).

## 2. Cài môi trường — Linux, Python 3.10

```bash
git clone --branch prod/version2 https://github.com/itschnmy/usseg_startracker.git
cd usseg_startracker
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: --require-hashes -r requirements-runtime.lock
bash scripts/materialize_assets.sh
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
Chi tiết: [báo cáo kiểm thử Podman](validation.md).

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
USSEG_IMAGE=localhost/usseg:prod-version2 USSEG_PYTHON=.venv/bin/python \
  bash scripts/build_podman.sh out/prod-release
USSEG_IMAGE=localhost/usseg:prod-version2 \
  bash scripts/run_test.sh out/demo-input out/demo-podman-output configs/runtime
```

Script chạy self-test rồi batch. Dùng input đã tạo ở bước 4 và output riêng cho
container. `configs/runtime/runtime.json` dùng đường dẫn container;
`development.json` dùng đường dẫn local.

Nếu nhận archive thay vì tự build:

```bash
cd out/prod-release
sha256sum -c SHA256SUMS
podman load -i usseg-prod-version2.oci.tar
cd ../..
```

Archive phải đúng kiến trúc host. Build hiện theo kiến trúc máy build;
archive amd64 không chạy native trên Jetson ARM64. Archive bàn giao là `usseg-prod-version2.oci.tar`.

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

Tài liệu tham chiếu: [thu thập ảnh Stellarium](../tools/stellarium/README.md).

## 7. Hợp đồng input/output và xử lý lỗi

### Metadata đầu vào

Ba file cùng stem: `frame.raw`, `frame.json`, `frame.ready`; payload encoded có
thể là `frame.bmp`/`frame.png` và được khai báo trong JSON. JSON schemas nằm tại
`schemas/`. Code kiểm tra thêm hash, giới hạn allocation, pathname và profile;
schema file không thay cho các kiểm tra cross-field đó.

Metadata tối thiểu:

```json
{
  "schema_version": "1.0",
  "job_id": "sample",
  "payload": "frame.raw",
  "payload_size_bytes": 614400,
  "payload_sha256": "REPLACE_WITH_SHA256",
  "camera_profile_id": "camera_raw12",
  "width": 640,
  "height": 480,
  "pixel_format": "Mono12",
  "processing_stage": "sensor_raw",
  "packing": "unpacked",
  "storage_bits": 16,
  "valid_bits": 12,
  "endianness": "little",
  "bit_alignment": "lsb",
  "stride_bytes": 1280,
  "header_bytes": 0,
  "trailer_bytes": 0
}
```

Ví dụ này phải thay SHA256 và tạo profile thật; không thể dùng nguyên cho camera
chưa xác định format. `.ready` là JSON chứa `schema_version`, `job_id`,
`metadata_sha256`. Producer ghi payload/metadata tạm, fsync và rename, rồi publish
ready cuối cùng. Sau ready tất cả file bất biến. Không đọc WCS/Corr/RA/Dec hints
qua metadata; field lạ bị reject. Symlink/traversal bị reject. Yêu cầu local POSIX
filesystem đáng tin cậy, producer tuân thủ protocol; chưa kiểm network filesystem.

| Format | Implementation | Điều kiện |
|---|---|---|
| encoded8 | PNG8/BMP grayscale decode | Header dimension khớp metadata/profile |
| Mono8/10/12 | Explicit unpacked, stride, header/trailer, little/big endian, LSB/MSB | Profile allowlist; golden synthetic đã test |
| BayerRG8/10/12 | Cùng unpacked adapter, RGGB tại origin, even dimensions | RGGB→gray; chưa có SDK/hardware sample |
| Packed `p`/`Packed` | `unsupported_format` | Chờ SDK sample/layout |
| ROI khác0, binning khác1, rotation/flip | Reject | Chờ adapter/profile calibration tương ứng |

Profiles kèm repo chỉ bật `encoded8`. RAW được test bằng byte fixtures tổng hợp;
không tự bật raw mode cho bay. `black_level/white_level` khóa theo profile, dark
native-depth optional được load một lần và kiểm hash/shape/range. Khi có dark,
master bao gồm offset nên không trừ black thêm lần nữa; scale denominator vẫn
white−black. Calibration float32 trước clip/rounduint8; Bayer demosaic grayscale
thực hiện sau fixed scaling. Chưa có distortion rectification/native-depth detector.
Input `already_corrected` yêu cầu calibration ID khớp, không cho dark/black subtraction
thêm. Runtime chưa xác minh exposure/temperature tương thích từ camera thật;
việc đó thuộc bước duyệt profile và protocol với hardware.

### Kết quả và mã thoát

```text
output/
  state/deployment.json
  state/active.json
  state/runner.lock
  runs/RUN_ID/
    run.json
    health.json
    results/MARKER_PATH_SHA256.json
    summary.json
    summary.csv
    events.jsonl
```

Result theo hash đường dẫn marker để job ID lỗi vẫn có key an toàn. Mỗi result
ghi schema/run/job IDs, input/hash/provenance, attempt/status và pose nullable.
Result được fsync+rename, rồi checkpoint mới clear. File partial `.pending-*`
không phải result và bị bỏ qua khi recovery; chúng có thể còn sau mất điện, cần
bên vận hành quản lý dung lượng. Result các attempt cũ được giữ ở run cũ.
`summary.statuses_latest` lấy attempt mới nhất theo marker; `statuses_this_run`
chỉ lượt này. Duplicate ID/hash giống bỏ qua có event; ID trùng/metadata khác
được lưu `invalid_input/job_id_conflict`. File đã ready mà sửa đổi là lỗi protocol.

| Status | Ý nghĩa |
|---|---|
| solved | Solver trả nghiệm, chưa có independent truth |
| no_solve | Pipeline chạy nhưng không tìm được nghiệm |
| invalid_input | Hỏng hash/metadata/geometry/calibration hoặc decode |
| unsupported_format | Format/packing không được hỗ trợ hoặc bật trong profile |
| timeout | Worker vượt hard deadline |
| runtime_error | Worker chết, lỗi thuật toán hoặc cạn retry |
| interrupted | Dừng quá grace, được retry có giới hạn |

Failure pose=null, không lặp lại attitude frame trước. Quaternion solved là
scalar-first `wxyz`, catalog→camera; +X boresight, +Y image-left, +Z image-up.
Pixel XY top-left; matched centroids YX trong nested algorithm output. Chưa có
mounting/body transform, covariance hay independent confidence calibration.
`mismatch_probability` là heuristic Tetra; không thay false-solve rate thực đo.

Exit0=batch xử lý xong không frame error; no_solve vẫn bình thường. Exit2=preflight;
exit3=batch có frame error; exit4=infrastructure/storage fatal; exit130/143=stop
bằng SIGINT/SIGTERM. Khi resume chỉ skip lỗi cũ thì exit có thể0; đọc
`summary.statuses_latest` nếu cần tình trạng tích lũy. Watch tiếp tục khi từng frame
lỗi; đọc health/results và các tổng trạng thái. Không chỉ dùng exit0 làm success rate.

Timing: decode gồm read/hash payload; calibration tách riêng; detector/ID/attitude
ở `timings_ns`. Worker total gồm adapter+algorithm. Supervisor frame gồm metadata,
startup/restart nếu có, chưa gồm output commit. Output-write latency nằm trong
`result_committed` event, gồm atomic result+checkpoint; không gọi đó là full
camera-to-host latency. Health có worker PID/progress; events được rotate một
file previous khi đạt budget. Summary có P50/P95/P99 và compute FPS; đây là timing
development, không cam kết realtime. `peak_rss_kib` là worker peak, không phải
peak cgroup/container. Results không tự xóa; free disk dưới ngưỡng dừng nhận job.
`max_ready_jobs` bảo vệ backlog; exceed thì fatal/backpressure, không drop ngầm.
