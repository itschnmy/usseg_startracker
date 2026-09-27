# usseg_startracker
Software module for CubeSat/Drone star tracker by USSEG

---

## 📚 Technical Documentation & System Design

Detailed system design, benchmark analysis, schemas, and sequence diagrams have been partitioned into modular technical specifications inside the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| **[System Architecture](docs/system_architecture.md)** | Architectural comparison between LOST (C++) and USSEG (Python), pipeline components, and submodule integration. |
| **[Benchmark Comparison](docs/benchmark_comparison.md)** | Empirical results and detailed evaluation across Synthetic LOST fixtures and DUST V2 real-world orbital flight imagery. |
| **[Data Dictionary & Schemas](docs/data_dictionary_and_schemas.md)** | Formal input/output schemas, pixel coordinates, star catalogs (BSC vs Hipparcos), and quaternion conventions. |
| **[Sequence Diagrams](docs/sequence_diagrams.md)** | Step-by-step sequence diagrams showing runtime execution flows for benchmarking, flight testing, and onboard operations. |
| **[Entity-Relationship Model](docs/entity_relationship.md)** | Relational data model and ER diagrams connecting scenarios, image frames, centroids, catalog stars, and metrics. |
| **[Branch Audit](docs/BRANCH_AUDIT.md)** | Selection rationale and historical integration audit of features across repository branches. |

---

## 🔗 Git Submodules Integration

This repository integrates upstream reference projects as Git submodules under `submodules/`:
*   **`submodules/lost`**: Upstream reference C++ star tracking implementation ([UWCubeSat/lost](https://github.com/UWCubeSat/lost.git)).
*   **`submodules/lost-evals`**: Upstream star tracking evaluation framework ([UWCubeSat/lost-evals](https://github.com/UWCubeSat/lost-evals.git)).

### Initializing Submodules:
When cloning this repository for the first time:
```bash
git clone --recurse-submodules https://github.com/itschnmy/usseg_startracker.git
```
Or initialize them in an existing checkout:
```bash
git submodule update --init --recursive
```

---

## 🚀 Unified Lost-In-Space Pipeline

The unified Python 3.10 pipeline processes camera images end-to-end through centroid detection, Tetra plate solving, and Wahba SVD attitude estimation.

```bash
cd /home/daniel/TGMT/usseg_startracker
source .venv/bin/activate
./scripts/materialize_assets.sh

python -m usseg_pipeline \
  --image submodules/lost-evals/scenarios-pyramid/20-low-noise/images/0.png \
  --database identificator/default_database.npz \
  --fov 20 \
  --output benchmark-results/example.json
```

The output JSON includes detected/matched star counts, passive and active quaternion conventions, plate-solver quality, per-stage nanosecond timings, total latency, and FPS. The bundled database covers 10–30° FOV.

### Generating the 45° Database:
```bash
curl -L -o identificator/hip_main.dat.gz \
  https://vizier.cfa.harvard.edu/ftp/cats/I/239/version_cd/cats/hip_main.dat.gz
.venv/bin/python scripts/generate_tetra_database.py \
  --catalog identificator/hip_main.dat.gz \
  --output identificator/database-fov45.npz --max-fov 45 --min-fov 45
```
*(Always use the complete 118,218-row CDS Hipparcos `I/239/hip_main.dat` catalog).*

### Environment Setup & Tests:
```bash
/home/daniel/.local/bin/uv venv --python 3.10 .venv
/home/daniel/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# Run pytest verification
.venv/bin/python -m pytest test/test_unified_pipeline.py
```

### Running Comparative LOST Fixtures:
```bash
cd submodules/lost-evals
source ../../.venv/bin/activate
USSEG_DIR=/home/daniel/TGMT/usseg_startracker \
  make SCENARIOS_PREFIX=scenarios-pyramid scenarios-pyramid/usseg.json
make SCENARIOS_PREFIX=scenarios-pyramid OUT_PREFIX=out compare-usseg
```

---

## 1. Python AttitudeDeterminator Package
Thư mục **`src/AttitudeDeterminator/`** chứa gói xử lý thái độ (Attitude Determination) được refactor hoàn toàn sang Python từ kịch bản C++ gốc kết hợp các thuật toán tiên tiến.

### Các thuật toán được hỗ trợ:
*   **TRIAD Estimator (`TRIADEstimator`):** Thuật toán xác định thái độ tĩnh từ 2 quan sát vector (Active Rotation, $Body \rightarrow Inertial$).
*   **QUEST Estimator (`QUESTEstimator`):** Giải toán tối ưu Wahba bằng phương pháp đa thức đặc trưng Shuster (Passive Rotation, $Inertial \rightarrow Body$).
*   **Davenport Q Estimator (`DavenportQEstimator`):** Giải tối ưu bằng cách tìm trị riêng lớn nhất của ma trận $K$ 4x4 (Passive Rotation, $Inertial \rightarrow Body$), khắc phục nhược điểm kỳ dị của QUEST tại góc xoay $180^\circ$.
*   **SVD Estimator (`SVDEstimator`):** Giải toán tối ưu Wahba bằng phân tích suy biến (Singular Value Decomposition), độ chính xác cao và số học ổn định tuyệt đối.
*   **MEKF Filter (`MEKFEstimator`):** Bộ lọc Kalman mở rộng nhân tính (6 trạng thái: 3 góc sai số thái độ, 3 sai số gyro bias). Hỗ trợ dự báo tần số cao bằng Gyro và cập nhật đo lường tuần tự bằng thuật toán **Murrell's Scheme** giúp tiết kiệm tài nguyên nhúng.

### Yêu cầu thư viện:
*   **`numpy`** (chỉ yêu cầu duy nhất thư viện NumPy để tính toán ma trận hiệu năng cao).

### Cách chạy kiểm thử Python:
Bạn có thể chạy kiểm thử toán học toàn bộ các bộ giải tĩnh và bộ lọc động MEKF thông qua dữ liệu chòm sao Crux thực tế bằng lệnh:
```bash
# Cài đặt thư viện nếu chưa có
pip install numpy

# Thực thi kịch bản test
python test/test_attitude_determinator.py
```

---

## 2. Dự án C++ gốc
Dành cho việc biên dịch và chạy các kiểm thử gốc bằng ngôn ngữ C++:

### Hướng dẫn biên dịch:
- **Bước 1:** Tạo cơ sở dữ liệu đồng bộ (Yêu cầu file `default_database.npz` trong thư mục `data/`):
```bash
python generate_kvec_db.py
```
- **Bước 2:** Build code C++ bằng CMake:
```bash
cd build
cmake ..
cmake --build .
```
- **Bước 3:** Chạy thử nghiệm LIS (chạy từ thư mục root của project):
```bash
cd ..
./build/test_lis
```
- **Bước 4:** Chạy thử nghiệm Tracking Mode:
```bash
./build/test_tracking
```

- **Bước 5:** Chạy thử nghiệm Tiền xử lý ảnh (Image Preprocessing - Yêu cầu OpenCV C++):
  Kiểm thử này yêu cầu thư viện OpenCV C++. Khi chạy CMake ở Bước 2, nếu hệ thống có OpenCV, target `test_image_preprocessing` sẽ tự động được kích hoạt và biên dịch.
  Để thực thi kiểm thử này từ thư mục root của project:
  ```bash
  ./build/test_image_preprocessing
  ```
  Sau khi chạy, kết quả ảnh lọc nền (`clean.png`), ảnh nhị phân băm lọc nhiễu (`binary.png`), và ảnh vẽ viền khoanh vùng ROI (`cropped_roi.png`) sẽ được xuất ra tại thư mục **`test/results/image_preprocessing/`** để kiểm tra trực quan.
