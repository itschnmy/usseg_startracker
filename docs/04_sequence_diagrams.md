<div align="center">

# Sơ Đồ Tuần Tự & Quy Trình Thực Thi Hệ Thống
### *Đặc Tả Quy Trình Thực Thi Runtime: Benchmark Giả Lập, Thẩm Định Dữ Liệu Bay & Vòng Lặp Bám Sao*

---

<!-- Navigation Bar -->
<p>
  <a href="../README.md#-tài-liệu-tiếng-việt"><img src="https://img.shields.io/badge/Trang_Chủ-README-blue?style=for-the-badge&logo=readme&logoColor=white" alt="README"/></a>
  &nbsp;&nbsp;
  <a href="README.md"><img src="https://img.shields.io/badge/Mục_Lục-Tài_Liệu_Docs-red?style=for-the-badge&logo=star&logoColor=white" alt="Docs"/></a>
</p>

---

</div>

Tài liệu này chi tiết hóa các bước thực thi qua các sơ đồ tuần tự (Sequence Diagrams) thể hiện quy trình luân chuyển dữ liệu và xử lý thời gian thực giữa các module: từ quy trình benchmark giả lập, quy trình thẩm định mù ảnh chuyến bay thực tế DUST V2, cho đến chu kỳ điều khiển bám sao tự động Lost-In-Space trên vệ tinh.

---

## 1. Quy Trình Thực Thi Benchmark Dữ Liệu Giả Lập

Sơ đồ này mô tả cách bộ điều phối benchmark nạp từng lô ảnh quang học xác định qua pipeline, trích xuất tâm sao, nhận dạng góc mẫu và đối chiếu với nghiệm chuẩn:

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
*Sơ đồ 8: Sơ đồ tuần tự đánh giá dữ liệu ảnh sao giả lập.*

---

## 2. Quy Trình Thẩm Định Mù Dữ Liệu Chuyến Bay Thực Tế DUST V2

Quy trình thẩm định mù (Blind Evaluation Protocol) trên ảnh bay vệ tinh CASSIOPE FAI: Các thuật toán thực hiện giải nghiệm hoàn toàn độc lập trước khi mở khóa cơ sở dữ liệu nghiệm kiểm chứng WCS và danh mục Tycho-2.

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
*Sơ đồ 9: Sơ đồ tuần tự thẩm định mù và đánh giá độc lập dữ liệu bay DUST V2.*

---

## 3. Chu Kỳ Điều Khiển Bám Sao Tự Động Lost-In-Space Trên Vệ Tinh

Quy trình mô tả chu kỳ thời gian thực tự động của hệ thống USSEG Star Tracker khi vận hành trên máy tính nhúng vệ tinh CubeSat hoặc máy bay không người lái UAV:

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
*Sơ đồ 10: Sơ đồ tuần tự chu kỳ bám sao tự động Lost-In-Space (LIS) tích hợp máy tính điều khiển ADCS.*
