# Version3 runtime — hướng dẫn chạy và hợp đồng draft1.0

Nhánh `dev/version3` đã có implementation batch/watch/self-test, worker watchdog,
ready protocol và persistent output. Config kèm repo là **profile lab**, Python3.10.
Architecture, camera SDK layout, quota và deadline bay vẫn chờ bên phần cứng.
Containerfile/scripts đã chuẩn bị; máy thực hiện lượt này chưa có Podman để build/run OCI.

## 1. Một lệnh test trên máy phát triển

Từ repo root, dùng Python3.10 đã có dependencies:

```bash
.venv/bin/python -m usseg_runtime.cli self-test \
  --config configs/runtime/development.json --output out/v3-local
```

Self-test chạy pipeline trên ảnh03, Stellarium, ảnh đen và payload lỗi, kiểm
attitude theo nhãn Stellarium sau inference, sau đó resume để kiểm không nhân đôi
result. Output nằm trong `out/v3-local/self-test/ID/`: `report.json`, `junit.xml`,
input và output từng frame. Exit0 khi checks pass; lỗi frame cố ý được tự kiểm.
Self-test này dùng known lab profiles; full pytest, C++ và DUST là test mặt đất riêng.

Môi trường runtime headless sạch (có `uv`):

```bash
uv venv --python .venv/bin/python .venv-runtime
uv pip install --python .venv-runtime/bin/python --require-hashes \
  --only-binary=:all: -r requirements-runtime.lock
.venv-runtime/bin/python -m usseg_runtime.cli self-test \
  --config configs/runtime/development.json --output out/v3-headless
```

Lock giữ versions của baseline, hash cho CPython3.10 Linux x86_64/aarch64 wheels.
Chưa chứng nhận ARM64 hardware. OCI base pin digest; cần xem xét dependency/license
và security maintenance trước release cuối, không tự nâng thư viện trong lúc bay.

## 2. Chuẩn bị input và chạy batch

Ví dụ publisher mặt đất cho encoded BMP/PNG:

```bash
.venv/bin/python -m usseg_runtime.cli prepare-input \
  --image dataset/200701.bmp --input out/v3-demo-input \
  --job-id sample200701 --profile lab640
.venv/bin/python -m usseg_runtime.cli batch \
  --input out/v3-demo-input --output out/v3-demo-output \
  --config configs/runtime/development.json
```

Publisher tạo thư mục mới, không ghi đè job cũ. JSON manifest ghi width/height,
payload size/hash và profile ID; thời gian capture không biết được giữ null.
Chạy lại batch với cùng output sẽ bỏ qua result terminal đã commit. Profile,
config, DB, source hoặc dependency/Python khác sẽ bị từ chối resume vào output cũ;
tạo output mới cho deployment mới. Không dùng output nằm trong input hoặc ngược lại.

Batch lấy snapshot các `.ready` có ở đầu lượt scan; file không có ready được chờ
đến lượt chạy sau. Watch nhận thêm jobs đến khi dừng:

```bash
.venv/bin/python -m usseg_runtime.cli watch \
  --input out/v3-demo-input --output out/v3-demo-output \
  --config configs/runtime/development.json
```

SIGTERM/SIGINT ngừng nhận việc, chờ frame hiện tại trong grace. Quá grace, worker
bị kill và result=`interrupted`; lần sau retry giới hạn `max_attempts`. SIGKILL
để lại active checkpoint; Linux parent-death signal kill worker và resume retry
job chưa có result. Nếu result đã commit mà checkpoint chưa clear, không xử lý lại.
Worker quá `frame_timeout_s` bị kill, frame=`timeout`; frame tiếp theo tạo worker mới.
Runtime hiện không tự retry các lỗi terminal khác; `interrupted` là trường hợp retry.

## 3. Input contract và RAW

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

## 4. Output và exit code

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

## 5. Build và chạy bằng Podman

Trên Linux host có Podman, sau khi checkout commit đã review và materialize DB:

```bash
USSEG_PYTHON=.venv/bin/python bash scripts/build_podman.sh out/podman-release
cd out/podman-release
sha256sum -c SHA256SUMS
podman load --input usseg-v3.oci.tar
```

Build script đòi working tree sạch để source revision có nghĩa; image name mặc
định `localhost/usseg:v3-dev`, override qua `USSEG_IMAGE`. Gói OCI vẫn là development
cho tới khi qua acceptance trên host đích. Không cần Docker. Build cần truy cập
registry/PyPI; runtime offline mang sẵn DB/dependencies/fixtures. `.containerignore`
loại dataset lớn, docs, GUI tools, `.git` và output khỏi context.

Chuẩn bị3 thư mục host. Config directory copy từ `configs/runtime/`; file
`runtime.json` trỏ DB cố định `/app/data/default_database.npz` trong image và
profile paths tương đối dưới `profiles/`. `development.json` dùng cho chạy từ repo.
Điền camera profile đã duyệt trước khi dùng ảnh camera bay.

```bash
# Đường dẫn ví dụ: host phải tạo và có quyền đọc/ghi phù hợp.
bash scripts/run_test.sh /srv/usseg/input /srv/usseg/output /srv/usseg/config
```

Script chạy self-test trên internal lab config rồi batch trên external runtime.json,
lưu output cùng host folder. Có `--pull=never`, network none, read-only root/input,
UID mapping keep-id và output writable; tmpfs256MiB là giá trị development.
Host SELinux phải cấu hình volume labels trước. CPU/memory/pids budget và stop
timeout phải chốt trên host đích; stop timeout cần lớn hơn grace cộng thời gian flush.

Watch dùng các option mount tương tự, command=`watch --config /config/runtime.json`.
Host đặt tên container và gọi `podman stop`; runtime nhận SIGTERM. Không chạy hai
container cùng output: lock sẽ từ chối. Rollback cần image/profile/output state
tương thích; khi source/dependency đổi dùng output deployment mới.

## 6. Quy trình kiểm chứng release

Ground suite tuần tự, cùng output folder mới:

```bash
bash scripts/verify_local.sh out/v3-verification-new
```

Script chạy Python tests → self-test → camera smoke → C++ LIS/tracking nếu có
compiler. C++ preprocessing chỉ chạy khi có OpenCV development headers/libs;
log ghi rõ skip. C++ là pipeline riêng, không thay kết quả Python camera.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest examples/tests -q
.venv-runtime/bin/python -m examples.run_version3_smoke --output out/v3-camera-new
```

Smoke publisher xử lý03/200701 raw/calibrated/Stellarium, xuất JSON và overlay chỉ
vẽ IDs solver trả về. Camera images thiếu WCS nên không tự gán correct. Cần thêm
SDK RAW sample, Podman build/run thật, đúng architecture, quota/OOM/power-loss,
cadence/soak và camera holdout trước ký nhận release bay.
