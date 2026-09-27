<div align="center">

# Đặc Tả Kiến Trúc Hệ Thống Star Tracker: LOST và USSEG
### *System Architecture Specification: LOST vs USSEG*

---

<!-- Navigation Bar -->
<p>
  <a href="../README.md#-tài-liệu-tiếng-việt"><img src="https://img.shields.io/badge/Trang_Chủ-README-blue?style=for-the-badge&logo=readme&logoColor=white" alt="README"/></a>
  &nbsp;&nbsp;
  <a href="README.md"><img src="https://img.shields.io/badge/Mục_Lục-Tài_Liệu_Docs-red?style=for-the-badge&logo=star&logoColor=white" alt="Docs"/></a>
</p>

---

</div>

Tài liệu này trình bày chi tiết thiết kế kiến trúc của cả hai pipeline bám sao **LOST** (C++) và **USSEG** (Python), sơ đồ tích hợp module, và phân tích so sánh chuyên sâu từng thành phần giải thuật trong chu trình xử lý.

---

## 1. So Sánh Kiến Trúc Tổng Thể Ở Mức Cao

Cả hai hệ thống đều giải quyết bài toán kinh điển **Mất Phương Hướng (Lost-In-Space - LIS)**: Cho một bức ảnh bầu trời sao chưa xác định chụp từ camera gắn trên vệ tinh, hệ thống tiến hành tách tọa độ tâm sao (centroid), nhận dạng các ngôi sao catalog thông qua so khớp mẫu hình học (Star-ID), và tính toán quaternion tư thế của vệ tinh đối với Hệ quy chiếu Thiên cầu Chuẩn (ICRF / ECI J2000).

```mermaid
graph LR
    subgraph S1["1. Raw Sensor Inputs"]
        direction TB
        S1_PNG["Synthetic / Flight PNG<br/>(8-bit Grayscale, 256x256)"]
        S1_H5["DUST V2 Level-1 HDF5<br/>(Spatial Slice 12:268)"]
    end

    subgraph S2["2. LOST Pipeline (C++ Core)"]
        direction TB
        S2_PRE["Image Normalization<br/>& Background Filter"]
        S2_DET["Centroiding Engine<br/>Center-of-Gravity (CoG)"]
        S2_FILTER["Centroid Filter<br/>Top 20 Brightest Stars"]
        S2_ID["Pyramid Star-ID<br/>K-Vector Distance Query"]
        S2_CAT["BSC Bright Star Catalog<br/>(mag le 5.0, 0.44 MB)"]
        S2_ATT["Attitude Estimator<br/>Davenport Q Method (DQM)"]
        S2_OUT["Active Quaternion<br/>Body to Inertial (ECI)"]
        S2_PRE --> S2_DET
        S2_DET --> S2_FILTER
        S2_FILTER --> S2_ID
        S2_CAT -.-> S2_ID
        S2_ID --> S2_ATT
        S2_ATT --> S2_OUT
    end

    subgraph S3["3. USSEG Pipeline (Python Core)"]
        direction TB
        S3_PRE["Adaptive Scaling<br/>Top-Hat Morphological Filter"]
        S3_DET["Centroiding Engine<br/>Connected Components (Area ge 1)"]
        S3_WRAP["Coordinate Mapper<br/>Image (x, y) to Tetra (y, x)"]
        S3_ID["Tetra Plate Solver<br/>4-Star Hash Table Lookup"]
        S3_CAT["Hipparcos Star Catalog<br/>(mag le 7.0, 47.1 MB)"]
        S3_ATT["Attitude Estimator<br/>Wahba SVD Optimal Solver"]
        S3_OUT["Passive Quaternion<br/>Inertial to Body"]
        S3_PRE --> S3_DET
        S3_DET --> S3_WRAP
        S3_WRAP --> S3_ID
        S3_CAT -.-> S3_ID
        S3_ID --> S3_ATT
        S3_ATT --> S3_OUT
    end

    subgraph S4["4. Evaluation & Verification"]
        direction TB
        S4_WCS["Astrometry.net WCS<br/>Pseudo-Ground-Truth"]
        S4_CORR["Tycho-2 Catalog<br/>Centroid Ground Truth"]
        S4_EVAL["Comparative Harness<br/>Boresight, Attitude & Timing"]
        S4_WCS -.-> S4_EVAL
        S4_CORR -.-> S4_EVAL
    end

    S1_PNG --> S2_PRE
    S1_H5 --> S2_PRE
    S1_PNG --> S3_PRE
    S1_H5 --> S3_PRE
    S2_OUT --> S4_EVAL
    S3_OUT --> S4_EVAL
```
*Hình 1: So sánh tổng quan giữa hai pipeline bám sao LOST và USSEG cùng hệ thống kiểm thử tự động.*

---

## 2. Phân Tích Chi Tiết Từng Giai Đoạn Trong Pipeline

### 2.1 Giai đoạn 1: Tiền Xử Lý Ảnh & Trích Xuất Tâm Sao (Centroid Extraction)

```mermaid
graph LR
    subgraph C1["LOST Centroiding Pipeline"]
        direction TB
        C1_IN["Input Raster Image"]
        C1_TH["Global Threshold Cutoff"]
        C1_CC["Connected Components & CoG"]
        C1_SORT["Flux Sort (Top 20 Brightest)"]
        C1_IN --> C1_TH
        C1_TH --> C1_CC
        C1_CC --> C1_SORT
    end

    subgraph C2["USSEG Centroiding Pipeline"]
        direction TB
        C2_IN["Input Raster Image"]
        C2_TOP["Morphological Top-Hat Filter"]
        C2_TH["Adaptive Threshold (mean + 3*sigma)"]
        C2_CC["Connected Components (Area ge 1)"]
        C2_SUB["Subpixel Center-of-Mass"]
        C2_SORT["Flux Ranking (Top 20 Stars)"]
        C2_IN --> C2_TOP
        C2_TOP --> C2_TH
        C2_TH --> C2_CC
        C2_CC --> C2_SUB
        C2_SUB --> C2_SORT
    end
```
*Hình 2: Luồng trích xuất tâm sao của LOST và USSEG.*

| Tiêu chí kỹ thuật | Pipeline của LOST | Pipeline của USSEG |
|---|---|---|
| **Ngôn ngữ triển khai** | C++14 / C++17 thuần | Python 3.10 (NumPy / SciPy / OpenCV) |
| **Chiến lược ngưỡng lọc nền** | Ngưỡng tĩnh hoặc cắt mức nền đơn giản | Lọc hình thái Top-Hat kết hợp ngưỡng thích nghi ($\mu + 3\sigma$) |
| **Độ nhạy hạt sao nhỏ (1-2 px)** | Bị lọc bỏ do ngưỡng diện tích lớn | **Nhận diện đầy đủ** (diện tích $\ge 1$ pixel, phù hợp cảm biến nhỏ $256 \times 256$) |
| **Độ chính xác tâm sao (Centroid)** | Trung bình ~0.814 px (PNG) / 1.034 px (H5) | **Sub-pixel cao ~0.346 px (PNG) / 0.502 px (H5)** |
| **Giới hạn số sao đầu ra** | Lấy 20 sao sáng nhất | Lấy 20 sao sáng nhất (được xếp hạng theo thông lượng quang thông) |

---

### 2.2 Giai đoạn 2: Nhận Dạng Sao Thiên Văn (Star Identification - Star-ID)

```mermaid
graph LR
    subgraph I1["LOST: Pyramid & K-Vector"]
        direction TB
        I1_VEC["Centroid Unit Vectors"]
        I1_TRI["Select Primary Triangle"]
        I1_KVEC["K-Vector Angular Query"]
        I1_CONF["4th Star Confirmation (Pyramid)"]
        I1_OUT["Identified Catalog Star IDs"]
        I1_VEC --> I1_TRI
        I1_TRI --> I1_KVEC
        I1_KVEC --> I1_CONF
        I1_CONF --> I1_OUT
    end

    subgraph I2["USSEG: Tetra Hash Matching"]
        direction TB
        I2_VEC["Centroid Unit Vectors"]
        I2_COMB["Generate 4-Star Combinations"]
        I2_HASH["Dimensionless Edge Invariants"]
        I2_LOOK["O(1) Hash Table Lookup"]
        I2_VERIF["Largest Clique Verification"]
        I2_OUT["Matched Catalog Vectors"]
        I2_VEC --> I2_COMB
        I2_COMB --> I2_HASH
        I2_HASH --> I2_LOOK
        I2_LOOK --> I2_VERIF
        I2_VERIF --> I2_OUT
    end
```
*Hình 3: Quy trình nhận dạng hình học giữa thuật toán Pyramid (LOST) và Tetra (USSEG).*

| Đặc tính giải thuật | LOST: Pyramid & K-Vector | USSEG: Tetra Hash Table |
|---|---|---|
| **Mô hình hình học** | Tam giác góc phẳng + Sao thứ 4 xác nhận kim tự tháp | Tứ giác 4 sao với tỷ lệ cạnh bất biến không thứ nguyên |
| **Độ phức tạp tra cứu** | $O(1)$ truy vấn K-Vector khoảng cách góc | $O(1)$ tra cứu trực tiếp trên bảng băm tứ giác |
| **Danh mục sao sử dụng** | Bright Star Catalog (BSC mag $\le 5.0$, 5.044 sao) | **Hipparcos CDS (mag $\le 7.0$, 118.218 sao)** |
| **Độ phủ catalog thiên cầu** | 49.08% (trong phạm vi 60 arcsec) | **86.97%** (dày đặc hơn gấp 1.77 lần) |
| **Khả năng chống giải sai** | Dễ bị false-positive do tứ giác ngẫu nhiên | **Kiểm tra đồ thị hoàn chỉnh (Clique Verification)** |

---

### 2.3 Giai đoạn 3: Ước Lượng Tư Thế Tối Ưu (Wahba Attitude Estimation)

Sau khi có danh sách các cặp vector đơn vị tương ứng giữa hệ quy chiếu camera cảm biến ($b_i$) và hệ quy chiếu thiên cầu quán tính ($r_i$), cả hai hệ thống tiến hành giải **Bài toán Wahba**:

$$\min_{R \in SO(3)} \frac{1}{2} \sum_{i=1}^N a_i \| b_i - R \, r_i \|^2$$

- **LOST sử dụng Phương pháp Davenport Q (DQM)**:
  - Chuyển đổi bài toán Wahba thành bài toán tìm trị riêng lớn nhất của ma trận $K_{4 \times 4}$.
  - Cho ra quaternion chủ động ($q_{active}: Body \rightarrow Inertial$).
- **USSEG sử dụng Phương pháp Phân tích Giá trị Kỳ dị (SVD)**:
  - Phân tích ma trận hiệp phát tán $B = \sum a_i b_i r_i^T = U S V^T$.
  - Nghiệm quay tối ưu: $R = U \operatorname{diag}(1, 1, \det(U)\det(V)) V^T$.
  - Ổn định số học tuyệt đối, không có điểm kỳ dị khi quay $180^\circ$, cho ra quaternion bị động ($q_{passive}: Inertial \rightarrow Body$).

---

## 3. Kiến Trúc Mô-đun Mã Nguồn Hiện Tại

```mermaid
graph LR
    subgraph M_CORE["Core Engine (models/)"]
        direction TB
        M_DET["models/detector<br/>Top-Hat + Connected Components"]
        M_ID["models/identifier<br/>Tetra Hash Table Solver"]
        M_ATT["models/attitude<br/>SVD, QUEST, Davenport Q, MEKF"]
        M_PIPE["models/pipeline<br/>Unified StarTrackerPipeline"]
        M_DET --> M_PIPE
        M_ID --> M_PIPE
        M_ATT --> M_PIPE
    end

    subgraph M_DATA["Data & Configs"]
        direction TB
        M_CONF["configs/<br/>Default Camera & Pipeline Presets"]
        M_CAT["data/<br/>Hipparcos Catalog & Tetra DB"]
        M_CONF --> M_CORE
        M_CAT -.-> M_ID
    end

    subgraph M_OUT["Documentation & Tests"]
        direction TB
        M_TEST["examples/tests<br/>Algorithmic Pytest Suite"]
        M_DOCS["docs/<br/>01 to 06 Technical Series"]
        M_PIPE --> M_TEST
        M_TEST --> M_DOCS
    end
```
*Hình 4: Thiết kế kiến trúc mô-đun hóa độc lập của thư viện USSEG.*
