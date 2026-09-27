# 🛰️ USSEG Star Tracker

<div align="center">

**Autonomous Embedded Star Tracker & Attitude Determination Engine**  
*Bộ Giải Biển Sao & Ước Lượng Tư Thế Vệ Tinh Nhúng cho CubeSat / Drone*

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

## 🎯 Current Status & Production Readiness

| Dimension | Current Achievement (v4) | Production Flight Target | Readiness Assessment |
|---|---|---|:---:|
| **Maturity Level** | **TRL 4–5** (Validated on Lab & Flight Data) | **TRL 7–8** (Flight-Ready Space Mission) | 🟡 **Pre-Production** |
| **Centroiding Accuracy** | **0.346 px** (PNG) / **0.502 px** (H5) | **< 0.15 px** across entire sensor | 🟢 **Surpassed LOST (2.3x)** |
| **Star Recall (Sensitivity)** | **38.43%** on noisy orbital frames | **> 30%** under high cosmic radiation | 🟢 **Surpassed LOST (7.5x)** |
| **Star-ID Precision** | **93.19%** true catalog star match | **> 90%** reliable matching | 🟢 **Production-Grade** |
| **Tracking Continuity** | **5 consecutive frames** ($0.26^\circ - 0.42^\circ$) | Continuous real-time track | 🟢 **Verified on Flight Data** |
| **Compute Latency (P50)** | **256 ms** (Python on x86_64) | **< 40 ms** (25+ FPS real-time on MCU) | 🟡 **Needs C++/Rust Port** |
| **Lens Optical Distortion** | Ideal pinhole camera model | Calibrated Brown-Conrady ($k_1, k_2, p_1, p_2$) | 🟡 **Needs Calibration LUT** |
| **Catalog Memory Footprint** | **47.1 MB** (NumPy NPZ format) | **< 3.5 MB** (Compact Flash memory) | 🟡 **Needs Bit-Quantization** |

> **Is it ready for Production?**  
> **Algorithmically: YES.** The detection, pattern matching, and Wahba SVD solver are battle-tested and outperform LOST on real orbital data.  
> **Flight-Hardware: NOT YET.** To deploy directly on a CubeSat flight computer (STM32H7 / ARM Cortex-M7), the core solver must be ported to C++/Rust, lens optical distortion must be calibrated, and the database compressed below 3.5 MB. See [06_roadmap_and_improvements.md](docs/06_roadmap_and_improvements.md).

---

## 📂 Repository Structure

```text
usseg_startracker/
├── configs/            # Preset parameters (FOV, thresholds, camera intrinsics)
├── data/               # Hipparcos catalog & pre-compiled Tetra databases (.npz, .db)
├── dataset/            # Sample orbital test frames & DUST fixtures
├── docs/               # Technical architecture & evaluation series (01 to 06)
├── examples/           # Demonstration scripts, DUST downloader & tests
│   ├── download_dust.py
│   ├── generate_tetra_database.py
│   └── tests/          # Algorithmic test suite
├── models/             # Core algorithmic modules (detector, identifier, attitude, pipeline)
│   └── cpp/            # Embedded C++ implementation & Eigen headers
├── utils/              # Mathematical utilities (SO(3), quaternions) & image processing
├── requirements.txt    # Python dependencies
└── README.md           # Master documentation
```

---

## 🌌 DUST V2 Flight Dataset: Download & Benchmark

The **DUST V2 (Dataset of Unknown Space Tracks)** is the official public orbital star tracker benchmark dataset (1,111 frames from the FAI NIR sensor).

### 1. Download & Extract:
```bash
# Automated download, MD5 verification (861 MB), and extraction from Zenodo
python examples/download_dust.py --destination dataset/DUST
```
*Manual Zenodo link:* [Zenodo Record 20255672](https://zenodo.org/records/20255672/files/DUST.zip?download=1) (MD5: `a31b62290eface15e519ed954124d59c`).

### 2. Dataset Structure & Formats:
- **`*.png`**: 8-bit grayscale flight images ($256 \times 256$).
- **`*.h5`**: Raw radiometric sensor matrix. Sliced at spatial ROI `[12:268, :]`.
- **`*.corr`**: Tycho-2 true star centroid coordinates for detector verification.
- **`*.wcs`**: Astrometry.net plate-solving headers for attitude ground truth.

### 3. Run Benchmark:
```bash
# Run USSEG on DUST V2 dataset
python -m models.pipeline   --image dataset/DUST/DUST/SET\ 1/2023_01_21/FAI_lv1_NIR_20230121_214442_214442_6.0.2.png   --database data/default_database.npz   --fov 26
```

---

## 📚 Technical Documentation Series

| # | Document | Overview |
|:---:|---|---|
| **01** | **[System Architecture](docs/01_system_architecture.md)** | End-to-end pipeline architecture, LOST vs USSEG component breakdown, and data flow. |
| **02** | **[Benchmark Comparison](docs/02_benchmark_comparison.md)** | Empirical evaluation across synthetic fixtures and 1,111 DUST V2 orbital flight frames. |
| **03** | **[Data Dictionary & Schemas](docs/03_data_dictionary_and_schemas.md)** | Input/output JSON schemas, pixel coordinates, catalogs (BSC vs Hipparcos), and quaternions. |
| **04** | **[Sequence Diagrams](docs/04_sequence_diagrams.md)** | Runtime call flows for benchmarking, blind flight image processing, and onboard LIS tracking. |
| **05** | **[Entity-Relationship Model](docs/05_entity_relationship.md)** | Relational model connecting scenarios, frames, centroids, matched stars, and solutions. |
| **06** | **[Roadmap & Improvements](docs/06_roadmap_and_improvements.md)** | **Production flight readiness**: optical distortion calibration, embedded C++ port, and flash compression. |

---

## 🚀 Quickstart

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run unit tests
pytest examples/tests

# 3. Test Lost-In-Space solve
python -m models.pipeline   --image dataset/9cee97e5-44a4-4193-9384-586386b7ab85.bmp   --database data/default_database.npz   --fov 26
```

---

<br/><br/>

---

<a id="-tài-liệu-tiếng-việt"></a>
# 🇻🇳 Tài Liệu Tiếng Việt

<p align="right"><a href="#-english-documentation">👉 Switch to English</a></p>

## 🎯 Hiện Trạng Dự Án & Mức Độ Sẵn Sàng Cho Production

| Tiêu chí | Đạt được hiện tại (v4) | Mục tiêu bay thực tế (Flight) | Đánh giá trạng thái |
|---|---|---|:---:|
| **Mức độ trưởng thành (TRL)** | **TRL 4–5** (Đã kiểm chứng trên lab & dữ liệu bay) | **TRL 7–8** (Sẵn sàng phóng vào vũ trụ) | 🟡 **Giai đoạn Tiền Production** |
| **Độ chính xác tâm sao (Centroid)** | **0.346 px** (PNG) / **0.502 px** (H5) | **< 0.15 px** trên toàn bộ cảm biến | 🟢 **Vượt trội LOST (gấp 2.3 lần)** |
| **Độ nhạy tách sao (Recall)** | **38.43%** trên ảnh quỹ đạo nhiều nhiễu | **> 30%** khi gặp bức xạ vũ trụ cao | 🟢 **Vượt trội LOST (gấp 7.5 lần)** |
| **Độ chuẩn xác Star-ID** | **93.19%** khớp chính xác sao catalog | **> 90%** nhận dạng ổn định | 🟢 **Đạt chuẩn Production** |
| **Bám đuôi liên tục (Tracking)** | **5 frames liên tiếp** ($0.26^\circ - 0.42^\circ$) | Duy trì liên tục theo thời gian thực | 🟢 **Đã kiểm chứng trên quỹ đạo** |
| **Thời gian tính toán (P50)** | **256 ms** (Python trên x86_64) | **< 40 ms** (25+ FPS trên vi điều khiển) | 🟡 **Cần viết lại C++/Rust nhúng** |
| **Méo quang học thấu kính** | Mô hình pinhole lý tưởng | Mô hình Brown-Conrady ($k_1, k_2, p_1, p_2$) | 🟡 **Cần nạp bảng hiệu chuẩn méo** |
| **Dung lượng cơ sở dữ liệu** | **47.1 MB** (Định dạng NPZ) | **< 3.5 MB** (Bộ nhớ Flash vi điều khiển) | 🟡 **Cần nén lượng tử hóa bit** |

> **Repo USSEG đã sẵn sàng cho Production chưa?**  
> **Về mặt Giải thuật: ĐÃ SẴN SÀNG.** Các thuật toán tách sao Top-Hat, nhận dạng hình học Tetra và giải thái độ Wahba SVD đều vượt trội thuật toán LOST trên dữ liệu bay thực tế.  
> **Về mặt Phần cứng nhúng: CẦN THÊM MỘT BƯỚC.** Để nạp trực tiếp vào máy tính điều khiển CubeSat (STM32H7 / ARM Cortex-M7), cần chuyển mã nguồn sang C++/Rust, tích hợp hiệu chuẩn méo thấu kính và nén cơ sở dữ liệu sao xuống dưới 3.5 MB. Chi tiết xem tại [06_roadmap_and_improvements.md](docs/06_roadmap_and_improvements.md).

---

## 📂 Cấu Trúc Thư Mục Chuẩn Hóa

```text
usseg_startracker/
├── configs/            # Cấu hình tham số (FOV, ngưỡng lọc, timeout...)
├── data/               # Danh mục sao Hipparcos & cơ sở dữ liệu Tetra (.npz, .db)
├── dataset/            # Ảnh quang học mẫu & dữ liệu DUST V2
├── docs/               # Hệ thống tài liệu kỹ thuật đánh số (01 đến 06)
├── examples/           # Scripts demo, bộ tải DUST & kịch bản kiểm thử
│   ├── download_dust.py
│   ├── generate_tetra_database.py
│   └── tests/          # Bộ unit tests xác minh thuật toán
├── models/             # Các mô-đun giải thuật chính (detector, identifier, attitude, pipeline)
│   └── cpp/            # Mã nguồn C++ truyền thống & thư viện Eigen
├── utils/              # Tiện ích toán học quay SO(3) & xử lý ảnh
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── README.md           # Hướng dẫn chính
```

---

## 🌌 Tập Dữ Liệu Quỹ Đạo Thực Tế DUST V2: Hướng Dẫn Tải & Xử Lý

**DUST V2 (Dataset of Unknown Space Tracks)** là bộ dữ liệu chuẩn gồm 1.111 ảnh chụp từ cảm biến FAI NIR trên quỹ đạo thực tế.

### 1. Tải và giải nén tự động:
```bash
# Tự động tải từ Zenodo (861 MB), kiểm tra mã băm MD5 và giải nén
python examples/download_dust.py --destination dataset/DUST
```
*Link Zenodo chính thức:* [Zenodo Record 20255672](https://zenodo.org/records/20255672/files/DUST.zip?download=1) (MD5: `a31b62290eface15e519ed954124d59c`).

### 2. Định dạng dữ liệu DUST:
- **`*.png`**: Ảnh hiển thị 8-bit ($256 \times 256$).
- **`*.h5`**: Dữ liệu cảm biến thô. Pipeline USSEG tự động cắt vùng không gian `[12:268, :]` để xử lý.
- **`*.corr`**: Tọa độ tâm sao thực tế (Tycho-2) dùng để đánh giá detector.
- **`*.wcs`**: Thông số WCS giải bởi Astrometry.net dùng làm chuẩn so sánh thái độ.

### 3. Chạy thử nghiệm trên ảnh DUST:
```bash
python -m models.pipeline   --image dataset/DUST/DUST/SET\ 1/2023_01_21/FAI_lv1_NIR_20230121_214442_214442_6.0.2.png   --database data/default_database.npz   --fov 26
```

---

## 📚 Hệ Thống Tài Liệu Kỹ Thuật Đánh Số

| Số hiệu | Tên tài liệu | Tóm tắt nội dung |
|:---:|---|---|
| **01** | **[Kiến trúc Hệ thống](docs/01_system_architecture.md)** | Thiết kế kiến trúc pipeline, so sánh thành phần LOST vs USSEG và luồng dữ liệu. |
| **02** | **[Báo cáo Benchmark](docs/02_benchmark_comparison.md)** | Kết quả thực nghiệm chi tiết trên ảnh giả lập và 1.111 khung ảnh quỹ đạo DUST V2. |
| **03** | **[Từ điển Dữ liệu & Schemas](docs/03_data_dictionary_and_schemas.md)** | Định dạng chuẩn JSON đầu ra, hệ tọa độ ảnh, danh mục sao và quy ước quaternion. |
| **04** | **[Sơ đồ Tuần tự Thực thi](docs/04_sequence_diagrams.md)** | Sơ đồ tuần tự các luồng runtime cho benchmark, xử lý ảnh bay và bám sao tự động. |
| **05** | **[Mô hình Thực thể Quan hệ](docs/05_entity_relationship.md)** | Mô hình ER kết nối kịch bản, khung ảnh, tâm sao, sao danh mục và nghiệm thái độ. |
| **06** | **[Lộ trình Nâng cấp & Sẵn sàng Production](docs/06_roadmap_and_improvements.md)** | **Định hướng bay không gian**: hiệu chuẩn méo thấu kính, viết lại C++ nhúng và nén Flash DB. |

---

## 🚀 Hướng Dẫn Nhanh (Quickstart)

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate

# 2. Chạy kiểm thử tự động
pytest examples/tests

# 3. Thực thi pipeline Lost-In-Space
python -m models.pipeline   --image dataset/9cee97e5-44a4-4193-9384-586386b7ab85.bmp   --database data/default_database.npz   --fov 26
```
