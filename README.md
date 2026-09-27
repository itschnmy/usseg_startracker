# 🛰️ USSEG Star Tracker

<div align="center">

**Embedded Lost-In-Space Star Tracker & Attitude Determination System**  
*Hệ thống Nhận dạng Sao & Xác định Tư thế Vệ tinh Nhúng cho CubeSat / Drone*

---

<!-- Language Switcher Bar -->
<p>
  <a href="#-english-documentation"><img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge&logo=google-chrome&logoColor=white" alt="English"/></a>
  &nbsp;&nbsp;
  <a href="#-tài-liệu-tiếng-việt"><img src="https://img.shields.io/badge/Ngôn_Ngữ-Tiếng_Việt-red?style=for-the-badge&logo=star&logoColor=white" alt="Tiếng Việt"/></a>
</p>

---

</div>

<br/>

<a id="-english-documentation"></a>
# 🇬🇧 English Documentation

<p align="right"><a href="#-tài-liệu-tiếng-việt">👉 Chuyển sang Tiếng Việt</a></p>

## 📌 Overview

**USSEG Star Tracker** is an autonomous, high-precision visual star tracking and attitude determination software suite designed for CubeSats, UAVs, and small satellite missions. It combines:
1. **Adaptive Morphological Star Detection**: Top-Hat filtering with connected-component clustering for sub-pixel centroid extraction under noisy orbital conditions.
2. **Geometric Hash Table Star Identification**: Tetra hash-table 4-star pattern matching for robust, rapid Lost-In-Space (LIS) solving.
3. **Wahba-Optimal Attitude Determination**: Deterministic static solvers (SVD, QUEST, Davenport Q, TRIAD) and a dynamic Multiplicative Extended Kalman Filter (MEKF) with Murrell's sequential measurement updates.

---

## 📂 Repository Structure

The codebase is organized into a clean, modular structure:

```text
usseg_startracker/
├── configs/            # Preset configurations (FOV, thresholds, camera intrinsics)
│   ├── __init__.py
│   └── default.py
├── data/               # Celestial catalogs & pre-compiled Tetra databases
│   ├── default_database.npz   # Bundled 10–30° FOV Tetra database
│   ├── database-fov45.npz     # 45° Wide-FOV Tetra database
│   ├── star_catalog.csv       # 5-star Crux benchmark catalog
│   └── kvector_fixed.db       # K-Vector database
├── dataset/            # Sample orbital test imagery & fixtures
│   └── 9cee97e5-44a4-4193-9384-586386b7ab85.bmp
├── docs/               # Modular numbered technical documentation
│   ├── 01_system_architecture.md
│   ├── 02_benchmark_comparison.md
│   ├── 03_data_dictionary_and_schemas.md
│   ├── 04_sequence_diagrams.md
│   ├── 05_entity_relationship.md
│   └── 06_branch_audit.md
├── examples/           # Demonstration scripts, database builders & OpenMV ports
│   ├── generate_tetra_database.py
│   ├── generate_kvec_db.py
│   ├── image_processing_openmv.py
│   └── tests/          # Algorithmic test suite
├── models/             # Core algorithmic pipelines & models
│   ├── detector/       # Star spot detection (Top-Hat + CC)
│   ├── identifier/     # Tetra plate solver
│   ├── attitude/       # Static solvers & MEKF attitude estimators
│   ├── pipeline/       # High-level unified Lost-In-Space pipeline
│   └── cpp/            # Legacy C++ embedded implementation & Eigen headers
├── utils/              # Mathematical utilities (SO(3), quaternions) & image processing
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📚 Technical Documentation (Numbered Series)

All technical reports and system architecture specifications are indexed below with direct links:

| # | Document | Focus Topics |
|---|---|---|
| **01** | **[System Architecture](docs/01_system_architecture.md)** | Architectural comparison between LOST (C++) and USSEG (Python), end-to-end pipeline diagrams, centroiding, Star-ID, and attitude determination. |
| **02** | **[Benchmark Comparison](docs/02_benchmark_comparison.md)** | Empirical evaluation across synthetic fixtures and DUST V2 real-world flight imagery (1,111 frames), timing, recall, and root cause analysis. |
| **03** | **[Data Dictionary & Schemas](docs/03_data_dictionary_and_schemas.md)** | Formal input/output JSON schemas, pixel coordinates, catalogs (BSC vs CDS Hipparcos), and quaternion conventions ($q_{active}$ vs $q_{passive}$). |
| **04** | **[Sequence Diagrams](docs/04_sequence_diagrams.md)** | Runtime execution sequence diagrams for synthetic benchmarking, blind orbital evaluation, and autonomous onboard LIS tracking. |
| **05** | **[Entity-Relationship Model](docs/05_entity_relationship.md)** | Relational data model connecting scenarios, image frames, centroids, matched stars, and attitude solution metrics. |
| **06** | **[Branch Audit](docs/06_branch_audit.md)** | Historical audit and rationale for consolidating features across repository branches. |

---

## 🚀 Quickstart: Unified Lost-In-Space Pipeline

Run the end-to-end star tracking pipeline on any image:

```bash
# Activate virtual environment
source .venv/bin/activate

# Execute pipeline via module
python -m models.pipeline   --image dataset/9cee97e5-44a4-4193-9384-586386b7ab85.bmp   --database data/default_database.npz   --fov 26   --attitude-method SVD
```

### Running Unit Tests:
```bash
.venv/bin/pytest -v examples/tests
```

---

## 📐 Attitude Determination Algorithms

The attitude determination suite (`models/attitude/`) provides:
*   **SVD Estimator (`SVDEstimator`)**: Wahba's problem solution using Singular Value Decomposition. Numerically robust, recommended for general LIS attitude determination.
*   **QUEST Estimator (`QUESTEstimator`)**: Shuster's quadratic eigenvalue approach for fast fine pointing.
*   **Davenport Q Estimator (`DavenportQEstimator`)**: Solves the Wahba eigenvalue problem on the $4 \times 4$ $K$-matrix without $180^\circ$ rotation singularities.
*   **TRIAD Estimator (`TRIADEstimator`)**: Fast deterministic two-vector solver.
*   **MEKF Filter (`MEKFEstimator`)**: Multiplicative Extended Kalman Filter tracking 6 states (3 attitude error angles, 3 gyro bias drifts) with Murrell's sequential measurement update scheme.

---

<br/><br/>

---

<a id="-tài-liệu-tiếng-việt"></a>
# 🇻🇳 Tài Liệu Tiếng Việt

<p align="right"><a href="#-english-documentation">👉 Switch to English</a></p>

## 📌 Giới Thiệu Tổng Quan

**USSEG Star Tracker** là bộ phần mềm định hướng và bám sao quang học tự động với độ chính xác cao, được thiết kế cho các sứ mệnh vệ tinh nhỏ (CubeSat), máy bay không người lái (UAV) và thiết bị vũ trụ. Hệ thống tích hợp:
1. **Nhận diện đốm sao thích nghi (Star Detection)**: Kết hợp biến đổi hình thái Top-Hat và gom cụm thành phần liên thông (Connected Components) giúp định vị tâm sao ở mức dưới pixel (sub-pixel) ngay cả trên ảnh chụp không gian có nhiều hạt nhiễu bức xạ.
2. **Nhận dạng hình học Tetra (Star Identification)**: Bảng băm hình học 4 ngôi sao (Tetra hash table) giải nhanh bài toán Mất Phương Hướng (Lost-In-Space) với tỷ lệ nhận dạng chính xác cao.
3. **Ước lượng thái độ tối ưu Wahba (Attitude Determination)**: Các bộ giải tĩnh tối ưu (SVD, QUEST, Davenport Q, TRIAD) kết hợp bộ lọc Kalman mở rộng nhân tính (MEKF) cập nhật tuần tự theo sơ đồ Murrell giúp tiết kiệm bộ nhớ trên vi điều khiển nhúng.

---

## 📂 Cấu Trúc Thư Mục Dự Án

Mã nguồn được tổ chức theo tiêu chuẩn mô-đun hóa hiện đại:

```text
usseg_startracker/
├── configs/            # Cấu hình tham số (FOV camera, ngưỡng sigma, timeout...)
│   ├── __init__.py
│   └── default.py
├── data/               # Danh mục sao thiên văn & cơ sở dữ liệu Tetra đã biên dịch
│   ├── default_database.npz   # Database mặc định FOV 10–30°
│   ├── database-fov45.npz     # Database góc rộng FOV 45°
│   ├── star_catalog.csv       # Danh mục sao thử nghiệm chòm Nam Thập Tự (Crux)
│   └── kvector_fixed.db       # Database K-Vector C++
├── dataset/            # Ảnh quang học quỹ đạo mẫu thử nghiệm
│   └── 9cee97e5-44a4-4193-9384-586386b7ab85.bmp
├── docs/               # Hệ thống tài liệu kỹ thuật được đánh số thứ tự
│   ├── 01_system_architecture.md
│   ├── 02_benchmark_comparison.md
│   ├── 03_data_dictionary_and_schemas.md
│   ├── 04_sequence_diagrams.md
│   ├── 05_entity_relationship.md
│   └── 06_branch_audit.md
├── examples/           # Kịch bản chạy mẫu, công cụ sinh database & code OpenMV
│   ├── generate_tetra_database.py
│   ├── generate_kvec_db.py
│   ├── image_processing_openmv.py
│   └── tests/          # Bộ kiểm thử thuật toán tự động
├── models/             # Các mô-đun giải thuật cốt lõi
│   ├── detector/       # Tách điểm sao (Top-Hat + Connected Components)
│   ├── identifier/     # Bộ giải biển sao Tetra (Plate Solver)
│   ├── attitude/       # Các bộ giải tư thế tĩnh & bộ lọc động MEKF
│   ├── pipeline/       # Pipeline hợp nhất xử lý Lost-In-Space
│   └── cpp/            # Mã nguồn C++ nhúng truyền thống & Eigen
├── utils/              # Tiện ích toán học quay không gian SO(3) & xử lý ảnh
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── README.md           # Hướng dẫn sử dụng chính
```

---

## 📚 Hệ Thống Tài Liệu Kỹ Thuật (Được đánh số thứ tự)

Các tài liệu kỹ thuật chi tiết đã được sắp xếp khoa học và có thể mở trực tiếp qua các liên kết bên dưới:

| Số hiệu | Tên tài liệu | Nội dung trọng tâm |
|:---:|---|---|
| **01** | **[Kiến trúc Hệ thống](docs/01_system_architecture.md)** | So sánh kiến trúc giữa LOST (C++) và USSEG (Python), sơ đồ luồng dữ liệu end-to-end, giải thuật trích xuất tâm sao, Star-ID và bộ giải thái độ. |
| **02** | **[Báo cáo So sánh Benchmark](docs/02_benchmark_comparison.md)** | Kết quả thực nghiệm trên ảnh giả lập và 1.111 frames dữ liệu quỹ đạo thực tế DUST V2, phân tích thời gian xử lý, độ nhạy và nguyên nhân sai số. |
| **03** | **[Từ điển Dữ liệu & Schemas](docs/03_data_dictionary_and_schemas.md)** | Đặc tả chuẩn dữ liệu đầu vào/đầu ra JSON, hệ tọa độ pixel, danh mục sao (BSC so với CDS Hipparcos), quy ước quaternion chủ động và bị động. |
| **04** | **[Sơ đồ Tuần tự Thực thi](docs/04_sequence_diagrams.md)** | Sơ đồ tuần tự các bước chạy thực nghiệm tổng hợp, quy trình đánh giá ảnh chuyến bay thực tế và chu kỳ bám sao tự động trên quỹ đạo. |
| **05** | **[Mô hình Thực thể Quan hệ](docs/05_entity_relationship.md)** | Mô hình dữ liệu quan hệ kết nối giữa các kịch bản thử nghiệm, khung ảnh, tọa độ tâm sao, sao danh mục và các chỉ số sai số thái độ. |
| **06** | **[Kiểm toán Lịch sử Nhánh](docs/06_branch_audit.md)** | Báo cáo kiểm toán lịch sử và căn cứ tích hợp các tính năng từ các nhánh phát triển trước đó. |

---

## 🚀 Hướng Dẫn Nhanh: Thực Thi Pipeline

Chạy thử nghiệm pipeline nhận diện sao trên một khung ảnh bất kỳ:

```bash
# Kích hoạt môi trường ảo Python
source .venv/bin/activate

# Chạy pipeline xử lý ảnh
python -m models.pipeline   --image dataset/9cee97e5-44a4-4193-9384-586386b7ab85.bmp   --database data/default_database.npz   --fov 26   --attitude-method SVD
```

### Chạy Bộ Kiểm Thử Thuật Toán (Pytest):
```bash
.venv/bin/pytest -v examples/tests
```

---

## 📐 Các Giải Thuật Ước Lượng Tư Thế

Gói thuật toán thái độ (`models/attitude/`) hỗ trợ đầy đủ các phương pháp:
*   **SVD Estimator (`SVDEstimator`)**: Giải bài toán Wahba bằng phân tích giá trị kỳ dị (SVD). Độ ổn định số học cao nhất, là phương pháp mặc định được khuyến nghị.
*   **QUEST Estimator (`QUESTEstimator`)**: Giải đa thức đặc trưng bậc 4 Shuster, tốc độ cao phù hợp bám đuôi mịn.
*   **Davenport Q Estimator (`DavenportQEstimator`)**: Tìm trị riêng lớn nhất của ma trận $K$ 4x4, loại bỏ hoàn toàn điểm kỳ dị góc xoay $180^\circ$.
*   **TRIAD Estimator (`TRIADEstimator`)**: Phương pháp giải định thức hai vector cổ điển, tính toán nhanh.
*   **Bộ lọc MEKF (`MEKFEstimator`)**: Bộ lọc Kalman mở rộng nhân tính ước lượng 6 trạng thái (3 góc sai số tư thế, 3 trôi lệch con quay hồi chuyển Gyro) với thuật toán cập nhật tuần tự Murrell giúp tối ưu hóa bộ nhớ RAM.
