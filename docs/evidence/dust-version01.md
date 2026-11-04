# LOST vs USSEG trên DUST V2

Kết quả test holdout theo session. H5 là track chính; PNG do DUST cung cấp là track tương thích. Correct chính: attitude error <0.5°, ngưỡng phụ <0.2°.

## Protocol đã khóa

- DUST archive MD5 `a31b62290eface15e519ed954124d59c`; 1378 frame, 22 session; test gồm 17 session/1.111 frame.
- Reference WCS hợp lệ trên 919 test frame; frame thiếu reference vẫn được tính solve-rate và timing.
- H5 float64 280×256 được chuyển về hình học FAI chính thức bằng `flipud(H5[12:268, :])`, rồi scale cố định từ dev: P0.1=0.0, P99.9=7335963.889934456; PNG dùng nguyên bản phát hành.
- Hai session `2023_06_03` và `2023_06_19` là ngoại lệ consistency H5/PNG theo geometry audit và được giữ lại để báo cáo minh bạch.
- Centroid caller dùng `(x,y)` zero-based; `PlateSolver` wrapper của USSEG tự đổi sang Tetra3 `(y,x)` bên trong.
- LOST được chấm trực tiếp bằng output inertial→camera; estimate nội bộ của USSEG được conjugate thành cùng chiều trước khi chấm.
- Cả hai algorithm chạy blind ở FOV 26°, một process/một thread; WCS/Corr chỉ được mở sau khi hai algorithm hoàn tất frame.

## Kết quả end-to-end

| Input | Algorithm | Frames | Solve micro/macro | Correct 0.5° micro/macro | Correct/solved | Correct 0.2° | Wrong solve | Attitude P50/P95 | Compute P50/P95 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H5 | LOST | 1111 | 17.822% / 18.932% | 0.109% / 0.085% | 0.610% | 0.000% | 17.737% | 128.843° / 172.005° | 95.696 ms / 520.979 ms | 7.271 |
| H5 | USSEG | 1111 | 1.620% / 1.636% | 0.109% / 0.115% | 5.556% | 0.000% | 1.850% | 32.798° / 112.476° | 362.605 ms / 614.672 ms | 2.862 |
| PNG | LOST | 1111 | 4.590% / 3.066% | 0.000% / 0.000% | 0.000% | 0.000% | 5.550% | 98.025° / 173.631° | 0.236 ms / 260.789 ms | 31.612 |
| PNG | USSEG | 1111 | 1.080% / 1.296% | 0.326% / 0.346% | 25.000% | 0.000% | 0.979% | 0.718° / 2.051° | 643.910 ms / 703.630 ms | 1.540 |

## Timing và tài nguyên

Compute loại trừ decode/adapter; end-to-end ở đây là decode + algorithm wall, chưa phải toàn bộ deployment latency. LOST wall bao gồm khởi động subprocess và đọc database, nhưng chưa gồm encoding/ghi PNG tạm. USSEG giữ database trong RAM. Peak RSS là mức process, không tách riêng hai algorithm.

| Input | Algorithm | Decode P50 | Compute P50/P95 | End-to-end P50/P95 | Compute/E2E FPS | Database size |
|---|---|---:|---:|---:|---:|---:|
| H5 | LOST | 9.858 ms | 95.696 ms / 520.979 ms | 146.929 ms / 703.686 ms | 7.271 / 4.642 | 0.443 MiB |
| H5 | USSEG | 9.858 ms | 362.605 ms / 614.672 ms | 392.873 ms / 641.697 ms | 2.862 / 2.635 | 47.125 MiB |
| PNG | LOST | 5.139 ms | 0.236 ms / 260.789 ms | 46.502 ms / 376.461 ms | 31.612 / 10.236 | 0.443 MiB |
| PNG | USSEG | 5.139 ms | 643.910 ms / 703.630 ms | 650.148 ms / 732.796 ms | 1.540 / 1.506 | 47.125 MiB |

## Detection và chuỗi thời gian

| Input | Algorithm | Labeled-star recall | Centroid P50 | Star-ID precision/recall | Continuity | Longest outage | Reacquisition mean |
|---|---|---:|---:|---:|---:|---:|---:|
| H5 | LOST | 19.465% | 1.034 px | — / — | 5.501% | 58.000 frames | 5.326 frames |
| H5 | USSEG | 19.097% | 0.502 px | 50.857% / 0.285% | 0.360% | 139.000 frames | 12.933 frames |
| PNG | LOST | 5.078% | 0.814 px | — / — | 1.369% | 139.000 frames | 4.276 frames |
| PNG | USSEG | 38.429% | 0.346 px | 93.485% / 0.444% | 0.346% | 128.000 frames | 20.750 frames |

## Phân tầng độ khó

Chi tiết đầy đủ nằm trong CSV `dust-usseg-vs-lost-summary-strata.csv`; các tercile background/sharpness chỉ là phân tích hậu nghiệm, không dùng để tuning.

| Input | Algorithm | Dimension | Group | Frames | Solve | Correct <0.5° | Attitude P50 |
|---|---|---|---|---:|---:|---:|---:|
| H5 | LOST | background_tercile | high | 381 | 19.423% | 0.000% | 124.458° |
| H5 | LOST | background_tercile | low | 343 | 12.536% | 0.315% | 119.726° |
| H5 | LOST | background_tercile | medium | 387 | 20.930% | 0.000% | 132.322° |
| H5 | LOST | exposure_s | 0.1 | 1111 | 17.822% | 0.109% | 128.843° |
| H5 | LOST | labelled_stars | 15-30 | 461 | 17.787% | 0.000% | 129.861° |
| H5 | LOST | labelled_stars | <15 | 8 | 25.000% | 0.000% | 112.281° |
| H5 | LOST | labelled_stars | >30 | 449 | 17.595% | 0.225% | 128.690° |
| H5 | LOST | labelled_stars | unknown | 193 | 18.135% | 0.000% | 86.358° |
| H5 | LOST | set | SET 1 | 711 | 16.034% | 0.171% | 127.961° |
| H5 | LOST | set | SET 2 | 400 | 21.000% | 0.000% | 129.727° |
| H5 | LOST | sharpness_tercile | low | 740 | 23.919% | 0.000% | 129.861° |
| H5 | LOST | sharpness_tercile | medium | 371 | 5.660% | 0.304% | 105.397° |
| H5 | USSEG | background_tercile | high | 381 | 1.312% | 0.000% | 2.201° |
| H5 | USSEG | background_tercile | low | 343 | 2.624% | 0.000% | 110.509° |
| H5 | USSEG | background_tercile | medium | 387 | 1.034% | 0.337% | 0.970° |
| H5 | USSEG | exposure_s | 0.1 | 1111 | 1.620% | 0.109% | 32.798° |
| H5 | USSEG | labelled_stars | 15-30 | 461 | 1.085% | 0.218% | 1.292° |
| H5 | USSEG | labelled_stars | <15 | 8 | 0.000% | 0.000% | — |
| H5 | USSEG | labelled_stars | >30 | 449 | 2.895% | 0.000% | 111.145° |
| H5 | USSEG | labelled_stars | unknown | 193 | 0.000% | 0.000% | — |
| H5 | USSEG | set | SET 1 | 711 | 1.266% | 0.171% | 1.292° |
| H5 | USSEG | set | SET 2 | 400 | 2.250% | 0.000% | 111.608° |
| H5 | USSEG | sharpness_tercile | low | 740 | 1.486% | 0.000% | 2.201° |
| H5 | USSEG | sharpness_tercile | medium | 371 | 1.887% | 0.304% | 63.394° |
| PNG | LOST | background_tercile | high | 381 | 0.000% | 0.000% | — |
| PNG | LOST | background_tercile | low | 343 | 14.869% | 0.000% | 98.025° |
| PNG | LOST | background_tercile | medium | 387 | 0.000% | 0.000% | — |
| PNG | LOST | exposure_s | 0.1 | 1111 | 4.590% | 0.000% | 98.025° |
| PNG | LOST | labelled_stars | 15-30 | 461 | 8.026% | 0.000% | 100.513° |
| PNG | LOST | labelled_stars | <15 | 8 | 0.000% | 0.000% | — |
| PNG | LOST | labelled_stars | >30 | 449 | 3.118% | 0.000% | 92.461° |
| PNG | LOST | labelled_stars | unknown | 193 | 0.000% | 0.000% | — |
| PNG | LOST | set | SET 1 | 711 | 5.767% | 0.000% | 105.491° |
| PNG | LOST | set | SET 2 | 400 | 2.500% | 0.000% | 61.904° |
| PNG | LOST | sharpness_tercile | high | 741 | 0.000% | 0.000% | — |
| PNG | LOST | sharpness_tercile | medium | 370 | 13.784% | 0.000% | 98.025° |
| PNG | USSEG | background_tercile | high | 381 | 1.312% | 0.000% | 0.782° |
| PNG | USSEG | background_tercile | low | 343 | 0.000% | 0.000% | — |
| PNG | USSEG | background_tercile | medium | 387 | 1.809% | 1.010% | 0.606° |
| PNG | USSEG | exposure_s | 0.1 | 1111 | 1.080% | 0.326% | 0.718° |
| PNG | USSEG | labelled_stars | 15-30 | 461 | 1.735% | 0.654% | 0.665° |
| PNG | USSEG | labelled_stars | <15 | 8 | 0.000% | 0.000% | — |
| PNG | USSEG | labelled_stars | >30 | 449 | 0.891% | 0.000% | 0.811° |
| PNG | USSEG | labelled_stars | unknown | 193 | 0.000% | 0.000% | — |
| PNG | USSEG | set | SET 1 | 711 | 0.985% | 0.512% | 0.606° |
| PNG | USSEG | set | SET 2 | 400 | 1.250% | 0.000% | 0.908° |
| PNG | USSEG | sharpness_tercile | high | 741 | 0.675% | 0.000% | 1.354° |
| PNG | USSEG | sharpness_tercile | medium | 370 | 1.892% | 0.955% | 0.606° |

## Oracle-centroid diagnostic

Corr centroid được đưa trực tiếp vào Tetra/attitude của USSEG; detector ảnh không tham gia và kết quả không được cộng vào end-to-end.

| Eligible Corr frames | Solve | Correct <0.5° all/solved | Attitude P50/P95 | Matches mean |
|---:|---:|---:|---:|---:|
| 918 | 27 (2.941%) | 0.436% / 14.815% | 1.129° / 2.628° | 9.630 |

## Kết luận trên code hiện tại

- LOST trên H5: solve 17.82%, correct <0.5° 0.11% toàn tập WCS / 0.610% trong các solve có WCS, wrong-solve 17.74%.
- USSEG trên H5: solve 1.62%, correct <0.5° 0.11% toàn tập WCS / 5.556% trong các solve có WCS, wrong-solve 1.85%.
- LOST trên PNG: solve 4.59%, correct <0.5° 0.00% toàn tập WCS / 0.000% trong các solve có WCS, wrong-solve 5.55%.
- USSEG trên PNG: solve 1.08%, correct <0.5° 0.33% toàn tập WCS / 25.000% trong các solve có WCS, wrong-solve 0.98%.
- Trên PNG, labeled-star recall của USSEG là 38.429% so với 5.078% của LOST; đây là phép đo detector, không tự động đồng nghĩa với attitude tốt hơn.
- Bỏ qua detector bằng Corr centroid, USSEG Tetra solve 2.94%; correct <0.5° là 0.44% toàn tập oracle / 14.815% trong các solve. Bottleneck vì vậy còn nằm ở solver/catalog/FOV-coordinate compatibility, không chỉ ở threshold detector.

## Cách diễn giải

- WCS được tạo bởi Astrometry.net nên là pseudo-ground-truth plate-solving, không phải đo attitude độc lập.
- Corr không bảo đảm liệt kê mọi sao nhìn thấy; unmatched-detection rate không đồng nghĩa hoàn toàn với false-positive rate.
- Star-ID agreement ở đây là overlap tập sao catalog theo RA/Dec trong 0.05°, chưa kiểm chứng từng cặp centroid–catalog. Không diễn giải nó thành correspondence-level precision.
- Outage/reacquisition tính theo số frame có sẵn, không phải số giây khi timestamp có khoảng trống.
- Telemetry Pitch/Roll/Yaw là spacecraft attitude; không được chấm như camera attitude vì DUST V2 không cung cấp FAI camera-to-body mounting transform.
- USSEG trong run này giữ nguyên detector uint8, pinhole lý tưởng và không dùng distortion/calibration DUST.
- Oracle-centroid chỉ có cho USSEG vì CLI hiện tại của LOST không nhận centroid ngoài.
