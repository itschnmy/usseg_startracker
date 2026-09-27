<div align="center">

# System Sequence Diagrams & Runtime Workflows
### *Sơ Đồ Tuần Tự & Quy Trình Thực Thi*

---

<!-- Language Switcher Bar -->
<p>
  <a href="../README.md#-english-documentation"><img src="https://img.shields.io/badge/Back_to-README-blue?style=for-the-badge&logo=readme&logoColor=white" alt="README"/></a>
  &nbsp;&nbsp;
  <a href="../README.md#-tài-liệu-tiếng-việt"><img src="https://img.shields.io/badge/Trang_Chủ-Tiếng_Việt-red?style=for-the-badge&logo=star&logoColor=white" alt="Tiếng Việt"/></a>
</p>

---

</div>

# Star Tracker Sequence Diagrams: Execution & Runtime Flows

This document details the step-by-step sequence diagrams showing the runtime and request flows for synthetic benchmarking, real-flight DUST V2 validation, and autonomous Lost-In-Space spacecraft operation.

---

## 1. Synthetic Benchmarking Execution Flow

This workflow illustrates how the evaluation harness executes a batch of deterministic synthetic star scenes through the candidate algorithm and measures accuracy against ground truth.

```mermaid
sequenceDiagram
    autonumber
    participant BR as Benchmark Runner
    participant AD as USSEG Pipeline Adapter
    participant SD as Star Detector
    participant PS as Tetra3 Plate Solver
    participant AE as Wahba SVD Estimator
    participant GT as Ground Truth Evaluator

    BR->>AD: run_scenario(image_path, fov, database_path)
    activate AD

    AD->>SD: detect_centroids(image_matrix)
    activate SD
    SD-->>AD: return centroid_list (x, y, flux)
    deactivate SD

    alt Centroids < 4
        AD-->>BR: return status: no_solve (insufficient stars)
    else Centroids >= 4
        AD->>PS: solve(centroid_list, fov, database)
        activate PS
        PS-->>AD: return matched_pairs (sensor_vec, catalog_vec)
        deactivate PS

        alt Star Identification Succeeded
            AD->>AE: estimate_attitude(sensor_vecs, catalog_vecs)
            activate AE
            AE-->>AD: return quaternion [w, x, y, z]
            deactivate AE
            AD-->>BR: return status: solved, quaternion, timings
        else Star Identification Failed
            AD-->>BR: return status: no_solve (pattern mismatch)
        end
    end
    deactivate AD

    BR->>GT: compute_angular_error(estimated_q, expected_q)
    activate GT
    GT-->>BR: return delta_theta_deg, correct_flag
    deactivate GT
    BR->>BR: record_metrics(timing, delta_theta)
```
*Figure 8: Sequence diagram for synthetic star image evaluation flow.*

---

## 2. Real-Flight DUST V2 Evaluation & Validation Flow

This workflow shows the blind evaluation protocol on CASSIOPE FAI flight images, where algorithms run completely blind before WCS pseudo-ground-truth and Tycho-2 catalogs are consulted.

```mermaid
sequenceDiagram
    autonumber
    participant Runner as DUST Test Runner
    participant Preproc as HDF5 Preprocessor
    participant Pipe as Star Tracker Pipeline
    participant Vault as WCS Ground Truth Vault
    participant Assessor as Performance Assessor

    Runner->>Preproc: load_frame(session_id, frame_idx)
    activate Preproc
    Preproc->>Preproc: crop_spatial_roi(spatial slice 12 to 268)
    Preproc->>Preproc: flip_vertical_np()
    Preproc->>Preproc: apply_scale(h5_scale_calibration)
    Preproc-->>Runner: return normalized_uint8_image
    deactivate Preproc

    Runner->>Pipe: solve_blind(normalized_uint8_image, fov=26.0)
    activate Pipe
    Pipe->>Pipe: extract_centroids()
    Pipe->>Pipe: match_catalog_patterns()
    Pipe->>Pipe: estimate_quaternion()
    Pipe-->>Runner: return solve_status, q_estimated, compute_time
    deactivate Pipe

    Runner->>Vault: unlock_ground_truth(session_id, frame_idx)
    activate Vault
    Vault-->>Runner: return wcs_quaternion, corr_star_positions
    deactivate Vault

    Runner->>Assessor: evaluate_frame(q_estimated, wcs_quaternion, corr_stars)
    activate Assessor
    Assessor->>Assessor: calculate_attitude_error(q_estimated, wcs_quaternion)
    Assessor->>Assessor: calculate_centroid_residuals(detected, corr_stars)
    Assessor->>Assessor: categorize_result(correct, wrong_solve, no_solve)
    Assessor-->>Runner: return frame_diagnostic_record
    deactivate Assessor
    Runner->>Runner: append_to_jsonl_and_summary()
```
*Figure 9: Sequence diagram for blind DUST V2 flight evaluation and validation protocol.*

---

## 3. Autonomous Onboard Lost-In-Space (LIS) Operational Cycle

This workflow illustrates the autonomous realtime flight cycle aboard a CubeSat or Drone Star Tracker running the unified USSEG package.

```mermaid
sequenceDiagram
    autonumber
    participant Camera as CMOS Image Sensor
    participant ImageProc as Preprocessor Module
    participant Centroid as Centroid Detector
    participant StarID as Tetra3 Star Matcher
    participant Attitude as Wahba SVD Solver
    participant ADCS as Spacecraft ADCS Computer

    ADCS->>Camera: trigger_exposure(exposure_ms=100)
    activate Camera
    Camera-->>ImageProc: dma_transfer_raw_pixels()
    deactivate Camera

    activate ImageProc
    ImageProc->>ImageProc: dark_frame_subtraction()
    ImageProc->>ImageProc: dynamic_range_quantization()
    ImageProc-->>Centroid: stream_processed_raster()
    deactivate ImageProc

    activate Centroid
    Centroid->>Centroid: compute_adaptive_threshold()
    Centroid->>Centroid: extract_contours_and_moments()
    Centroid-->>StarID: pass_top_20_centroids()
    deactivate Centroid

    activate StarID
    StarID->>StarID: form_4_star_hash_combinations()
    StarID->>StarID: query_hipparcos_hash_table()
    StarID-->>Attitude: pass_matched_star_vectors()
    deactivate StarID

    activate Attitude
    Attitude->>Attitude: compute_svd_rotation_matrix()
    Attitude->>Attitude: matrix_to_quaternion()
    Attitude->>Attitude: verify_residual_variance()
    Attitude-->>ADCS: telemetry_packet(quaternion, status=VALID)
    deactivate Attitude

    ADCS->>ADCS: update_extended_kalman_filter(quaternion)
    ADCS->>ADCS: command_reaction_wheels()
```
*Figure 10: Sequence diagram for autonomous onboard Lost-in-Space (LIS) cycle.*
