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
*Figure 1: High-level comparison of LOST and USSEG end-to-end star tracking pipelines and evaluation harness.*

---

## 2. Pipeline Stage Breakdown

### 2.1 Stage 1: Preprocessing & Centroid Extraction

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
