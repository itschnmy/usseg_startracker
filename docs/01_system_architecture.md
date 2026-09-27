<div align="center">

# System Architecture Specification
### *Đặc Tả Kiến Trúc Hệ Thống*

---

<!-- Language Switcher Bar -->
<p>
  <a href="../README.md#-english-documentation"><img src="https://img.shields.io/badge/Back_to-README-blue?style=for-the-badge&logo=readme&logoColor=white" alt="README"/></a>
  &nbsp;&nbsp;
  <a href="../README.md#-tài-liệu-tiếng-việt"><img src="https://img.shields.io/badge/Trang_Chủ-Tiếng_Việt-red?style=for-the-badge&logo=star&logoColor=white" alt="Tiếng Việt"/></a>
</p>

---

</div>

# Star Tracker System Architecture: LOST vs USSEG

This document presents the detailed architectural design of both the **LOST** (C++) and **USSEG** (Python) star tracker pipelines, their submodule integrations, and a comparative analysis of each pipeline component.

---

## 1. High-Level Architectural Comparison

Both systems solve the classic **Lost-In-Space (LIS)** problem: given an unidentified star field image taken by an onboard camera, detect star centroids, identify catalog stars by matching star patterns, and compute the spacecraft/camera attitude quaternion with respect to the Celestial Reference Frame (ICRF / ECI J2000).

```mermaid
flowchart TD
    subgraph Inputs["1. Raw Sensor Inputs"]
        RAW_PNG["Synthetic or Flight PNG Image (8-bit grayscale)"]
        RAW_H5["DUST V2 FAI Level-1 HDF5 (spatial slice 12:268)"]
    end

    subgraph LOST_Pipe["2. LOST Pipeline (C++ Core)"]
        L_PRE["Image Normalization & Background Filter"]
        L_DET["Centroiding: Center-of-Gravity (CoG)"]
        L_FILTER["Brightness Filter (Top 20 Stars)"]
        L_ID["Star ID: Pyramid Algorithm + K-Vector Search"]
        L_CAT[("BSC Catalog: V-mag le 5.0, 0.44 MB")]
        L_ATT["Attitude Solver: Davenport Q Method (DQM)"]
        L_OUT["LOST Quaternion (Active Body to Inertial)"]
    end

    subgraph USSEG_Pipe["3. USSEG Pipeline (Python Core)"]
        U_PRE["Dynamic Scaling (h5-scale dev calibration)"]
        U_DET["Adaptive Thresholding (mean + 3*sigma) & Contour Centroiding"]
        U_WRAP["Coordinate Mapper (x,y zero-based to Tetra3 y,x)"]
        U_ID["Plate Solver: Tetra3 4-Star Hash Matching"]
        U_CAT[("Hipparcos Catalog: V-mag le 7.0, 47.1 MB")]
        U_ATT["Attitude Solver: Wahba SVD Estimator"]
        U_CONV["Conjugate Conversion for LOST/ECI Alignment"]
        U_OUT["USSEG Quaternion (Passive Inertial to Body)"]
    end

    subgraph Eval_Harness["4. Unified Evaluation Harness"]
        HARNESS["Evaluation Runner & Ground Truth Assessor"]
        WCS_REF[("Astrometry.net WCS Pseudo-Ground-Truth")]
        CORR_REF[("Tycho-2 Corr Star Catalog")]
        METRICS["Metric Evaluator: Availability, Solve Rate, Precision, Latency"]
    end

    RAW_PNG --> L_PRE
    RAW_H5 --> L_PRE
    RAW_PNG --> U_PRE
    RAW_H5 --> U_PRE

    L_PRE --> L_DET --> L_FILTER --> L_ID
    L_CAT -.-> L_ID
    L_ID --> L_ATT --> L_OUT

    U_PRE --> U_DET --> U_WRAP --> U_ID
    U_CAT -.-> U_ID
    U_ID --> U_ATT --> U_CONV --> U_OUT

    L_OUT --> HARNESS
    U_OUT --> HARNESS
    WCS_REF -.-> HARNESS
    CORR_REF -.-> HARNESS
    HARNESS --> METRICS
```
*Figure 1: High-level comparison of LOST and USSEG end-to-end star tracking pipelines and evaluation harness.*

---

## 2. Pipeline Stage Breakdown

### 2.1 Stage 1: Preprocessing & Centroid Extraction

```mermaid
flowchart LR
    subgraph LOST_Centroid["LOST Centroiding Flow"]
        L1["Input Raster"] --> L2["Threshold Cutoff"]
        L2 --> L3["Connected Component CoG"]
        L3 --> L4["Flux Sort (Top 20 Brightest)"]
    end

    subgraph USSEG_Centroid["USSEG Centroiding Flow"]
        U1["Input Raster"] --> U2["Adaptive Threshold (mean + 3*sigma)"]
        U2 --> U3["Contour Extraction (Min Area 2)"]
        U3 --> U4["Subpixel Center-of-Mass"]
        U4 --> U5["Flux Ranking (Top 20 Stars)"]
    end
```
*Figure 2: Centroid detection workflows in LOST and USSEG.*

| Dimension | LOST Pipeline | USSEG Pipeline |
|---|---|---|
| **Implementation Language** | C++14 / C++17 | Python 3.10 (NumPy / SciPy / OpenCV) |
| **Algorithm** | Center-of-Gravity (CoG) with thresholding (`--centroid-algo=cog`) | Global threshold `mean + 3*sigma`, 8-connectivity contour bounding, local background subtraction |
| **Dynamic Range Handling** | Linear fixed offset and scale | Dynamic percentile scaling (`h5-scale.json` locked from dev set) |
| **Centroid Selection** | Selects brightest $N$ peaks (default 20 on flight frames) | Contours sorted by integrated flux, top 20 candidate centroids |
| **Subpixel Precision** | Intensity-weighted centroid formula around local peak | 2D image moment center of mass within connected component |
| **Coordinate System** | Zero-based $(x, y)$ column/row indices | Zero-based $(x, y)$, converted internally to $(y, x)$ for Tetra3 |

### 2.2 Stage 2: Star Identification (Star-ID)

```mermaid
flowchart LR
    subgraph LOST_ID["LOST Pyramid / K-Vector Flow"]
        LP1["Centroid Vectors"] --> LP2["Pick Primary Triangle"]
        LP2 --> LP3["K-Vector Angular Distance Query"]
        LP3 --> LP4["Validate 4th Star (Pyramid Confirmation)"]
        LP4 --> LP5["Matched Star ID Set"]
    end

    subgraph USSEG_ID["USSEG Tetra3 Hash Flow"]
        UP1["Centroid Vectors"] --> UP2["Generate 4-Star Combinations"]
        UP2 --> UP3["Compute Dimensionless Edge Hashes"]
        UP3 --> UP4["O(1) Hash Table Lookup (Hipparcos)"]
        UP4 --> UP5["Largest Clique Verification"]
    end
```
*Figure 3: Star pattern recognition algorithms: LOST Pyramid vs USSEG Tetra3.*

| Dimension | LOST Pipeline | USSEG Pipeline |
|---|---|---|
| **Core Algorithm** | Pyramid Algorithm (Mortari et al.) accelerated by K-Vector table search | Tetra3 Hash-based 4-Star Combination Matching (ESA Tetra adaptation) |
| **Catalog** | Bright Star Catalog (BSC5) | Hipparcos Catalog (`hip_main.dat`, CDS I/239 complete 118,218 stars) |
| **Limiting Magnitude** | Magnitude $\le 5.0$ (sparse, optimized for low memory) | Magnitude $\le 7.0$ (dense, robust for small fields of view) |
| **Database Size** | ~0.443 MiB (for 26° FOV) | ~47.125 MiB (10°–30° multiscale) / ~3.14 MiB (45° single-scale) |
| **In-Memory Query Cost** | $O(1)$ range lookup via K-Vector | $O(1)$ hash table lookup of angular separation hash keys |
| **Failure Mode** | Returns no solve if insufficient pyramid triangles match | Early exit if no valid 4-star pattern matches hash catalog |

### 2.3 Stage 3: Attitude Determination

| Dimension | LOST Pipeline | USSEG Pipeline |
|---|---|---|
| **Algorithm** | Davenport Q Method (DQM) | Singular Value Decomposition (SVD) of Wahba Problem |
| **Loss Function** | Maximizes Wahba gain via quaternion eigenvalue decomposition | Minimizes weighted sum of squared vector residual errors via SVD |
| **Attitude Representation** | Unit Quaternion $\mathbf{q} = [w, x, y, z]$ (Scalar first) | Unit Quaternion $\mathbf{q} = [w, x, y, z]$ (Scalar first) |
| **Frame Convention** | Active rotation from camera body frame to inertial frame | Passive rotation from inertial frame to camera body frame |
| **Alignment Layer** | Native LOST output | Conjugate inversion layer (`quaternion_wxyz = conj(q_passive)`) |

---

## 3. Submodule Integration Architecture

To maintain maximum architectural modularity and allow reproducible comparative benchmarking, both `lost` and `lost-evals` are integrated into `usseg_startracker` via standard Git submodules under `submodules/`.

```mermaid
flowchart TD
    subgraph Repo["usseg_startracker Architecture"]
        direction TB
        CORE["models/attitude/ (SVD, QUEST, Davenport Q, TRIAD, MEKF)"]
        DET["models/detector/ (Top-Hat + Connected Components)"]
        ID["models/identifier/ (Tetra Plate Solver)"]
        PIPE["models/pipeline/ (Unified Pipeline & CLI)"]
        CONFIG["configs/ (Default Parameters & Presets)"]
        DATA["data/ (Catalogs & Tetra Database)"]
        DOCS["docs/ (01 to 06 Numbered Documentation)"]
        TESTS["examples/tests/ (Algorithmic Verification)"]
    end

    DET --> PIPE
    ID --> PIPE
    CORE --> PIPE
    CONFIG --> PIPE
    DATA -.-> ID
    PIPE --> TESTS
    TESTS --> DOCS
```
*Figure 4: Submodule layout and dependency flow within `usseg_startracker`.*

- **`submodules/lost`**: Tracks the official upstream C++ implementation (`https://github.com/UWCubeSat/lost.git`). Used to compile the reference `lost` binary CLI.
- **`submodules/lost-evals`**: Tracks the official evaluation framework (`https://github.com/UWCubeSat/lost-evals.git`). Contains the Monte Carlo generation scripts and evaluation pipelines.
- **`usseg_pipeline`**: The production-ready unified Python package providing a clean CLI (`python -m usseg_pipeline`) and programmatic API (`StarTrackerPipeline`) consumed by evaluation runners.

---

## 4. Key Architectural Trade-offs

1. **Memory Footprint vs Star Identification Density**:
   - LOST prioritizes ultra-low memory (~443 KB database, BSC mag $\le 5.0$), allowing it to fit into microcontrollers with constrained RAM. However, on faint flight images or small FOV sensors, fewer stars are visible, lowering recall.
   - USSEG utilizes the complete Hipparcos catalog down to magnitude 7.0 (~47 MB database), ensuring high all-sky star density (86.97% catalog coverage in 60 arcsec), but requiring more memory and longer search times when noise is present.

2. **Speed vs False Positive Immunity**:
   - LOST achieves higher FPS (>100 FPS compute) and lower latency (~3-8 ms compute) on low-noise synthetic images. However, when faced with high noise or flight blur, its pyramid solver can yield wrong attitudes (8% wrong solves in 45° high noise, 17.74% wrong solves on DUST H5).
   - USSEG’s 4-star Tetra3 hash matching enforces strict geometric constraints. In all synthetic pilot runs, USSEG yielded **0% wrong solves** (it prefers a clean `no_solve` early exit rather than corrupting spacecraft navigation with a false attitude).
