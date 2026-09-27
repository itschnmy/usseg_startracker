# Star Tracker Entity-Relationship Diagram & Data Model

This document defines the relational data model connecting benchmark scenarios, image frames, detected centroids, catalog stars, identified matches, attitude solutions, and evaluation metrics across the star tracking system.

---

## 1. Entity-Relationship Diagram

```mermaid
erDiagram
    Scenario ||--o{ ImageFrame : contains
    ImageFrame ||--o{ Centroid : detects
    ImageFrame ||--o| GroundTruthReference : has
    ImageFrame ||--o| AttitudeSolution : produces
    CatalogStar ||--o{ StarMatch : identifies
    Centroid ||--o{ StarMatch : matches
    AttitudeSolution ||--o{ StarMatch : utilizes
    BenchmarkRun ||--o{ FrameEvaluationRecord : records
    ImageFrame ||--o{ FrameEvaluationRecord : evaluated_by
    AttitudeSolution ||--o| FrameEvaluationRecord : assessed_in

    Scenario {
        string scenario_id PK
        string scenario_name
        float fov_deg
        string noise_profile
        int total_frames
    }

    ImageFrame {
        string frame_id PK
        string scenario_id FK
        string file_path
        int width_px
        int height_px
        float exposure_time_s
        string format
    }

    Centroid {
        string centroid_id PK
        string frame_id FK
        float x_px
        float y_px
        float flux_adu
        float snr
        int rank
    }

    CatalogStar {
        int catalog_id PK
        string catalog_name
        float ra_deg
        float dec_deg
        float visual_magnitude
        float unit_vector_x
        float unit_vector_y
        float unit_vector_z
    }

    StarMatch {
        string match_id PK
        string centroid_id FK
        int catalog_id FK
        string solution_id FK
        float angular_residual_deg
    }

    AttitudeSolution {
        string solution_id PK
        string frame_id FK
        string algorithm_name
        string status
        float quat_w
        float quat_x
        float quat_y
        float quat_z
        float compute_time_ms
        int matched_stars_count
    }

    GroundTruthReference {
        string reference_id PK
        string frame_id FK
        float true_quat_w
        float true_quat_x
        float true_quat_y
        float true_quat_z
        string source_type
    }

    BenchmarkRun {
        string run_id PK
        string run_timestamp
        string environment_info
        string code_commit_hash
        string runner_name
    }

    FrameEvaluationRecord {
        string eval_id PK
        string run_id FK
        string frame_id FK
        string solution_id FK
        float attitude_error_deg
        boolean is_correct
        boolean is_wrong_solve
        boolean is_no_solve
        float centroid_residual_px
    }
```
*Figure 11: Entity-relationship diagram for star tracker benchmarking and execution data model.*

---

## 2. Entity Dictionary & Attributes

### 2.1 `Scenario`
Represents an evaluation configuration or simulation envelope.
- `scenario_id` (PK, string): Unique identifier (e.g. `20-low-noise`, `dust-v2-test`).
- `fov_deg` (float): Camera field-of-view in degrees.
- `noise_profile` (string): Noise characteristics (`ideal`, `low_gaussian`, `flight_auroral`).
- `total_frames` (int): Number of frames in the scenario.

### 2.2 `ImageFrame`
An individual camera exposure or simulated raster.
- `frame_id` (PK, string): Frame identifier (e.g. `0.png`, `2023_06_19_001.h5`).
- `file_path` (string): Relative or absolute filesystem path.
- `width_px`, `height_px` (int): Pixel dimensions.
- `exposure_time_s` (float): Exposure duration in seconds.

### 2.3 `Centroid`
A candidate star spot extracted by the image detector.
- `centroid_id` (PK, string): Unique detection identifier.
- `x_px`, `y_px` (float): Subpixel image coordinates (zero-based).
- `flux_adu` (float): Integrated pixel energy above background.
- `snr` (float): Signal-to-noise ratio.
- `rank` (int): Brightness ranking among candidate spots in the frame.

### 2.4 `CatalogStar`
A reference celestial object from a standardized catalog.
- `catalog_id` (PK, int): Identifier (BSC number or Hipparcos HIP number).
- `catalog_name` (string): Originating catalog (`BSC5` or `Hipparcos`).
- `ra_deg`, `dec_deg` (float): Celestial coordinates (Right Ascension / Declination) in J2000 epoch.
- `unit_vector_x/y/z` (float): Precomputed 3D unit coordinates on Celestial Sphere.

### 2.5 `StarMatch`
An association between a detected 2D centroid and a 3D catalog star established by pattern recognition.
- `match_id` (PK, string): Match identifier.
- `angular_residual_deg` (float): Angular separation between rotated sensor vector and true catalog vector.

### 2.6 `AttitudeSolution`
The estimated attitude quaternion and runtime telemetry produced by the pipeline.
- `solution_id` (PK, string): Solution instance identifier.
- `status` (string): Result state (`solved`, `no_solve`, `timeout`, `error`).
- `quat_w`, `quat_x`, `quat_y`, `quat_z` (float): Unit quaternion components (scalar-first).
- `compute_time_ms` (float): Execution latency of algorithm stages in milliseconds.

### 2.7 `GroundTruthReference`
The verified reference attitude and star field truth.
- `reference_id` (PK, string): Reference record identifier.
- `true_quat_w/x/y/z` (float): Reference quaternion.
- `source_type` (string): Origin (`simulator_exact`, `astrometry_wcs_pseudotruth`).

### 2.8 `FrameEvaluationRecord`
The audit outcome comparing a pipeline's `AttitudeSolution` against the `GroundTruthReference`.
- `eval_id` (PK, string): Audit record identifier.
- `attitude_error_deg` (float): Angular geodesic distance between solution and ground truth:
  $$\Delta\theta = 2 \arccos(|\mathbf{q}_{est} \cdot \mathbf{q}_{true}|)$$
- `is_correct` (boolean): `true` if $\Delta\theta < 0.5^\circ$.
- `is_wrong_solve` (boolean): `true` if solved, but $\Delta\theta \ge 0.5^\circ$.
- `is_no_solve` (boolean): `true` if algorithm returned no solution.
