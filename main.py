import os
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CẤU HÌNH & THUẬT TOÁN TỪ FILE ROTATION.PY
# ============================================================
CELL_SIZE = 100.0
MAX_SEARCH_RADIUS = 150.0
INTENSITY_TOLERANCE = 10.0
NN_RATIO_TEST = 0.90
RANSAC_ITERATIONS = 1000
INLIER_THRESHOLD = 1.5
EARLY_STOP_INLIER_RATIO = 1.0


def build_spatial_hash(points, cell_size: float):
    """Xây dựng bảng băm không gian (Spatial Hash Grid) từ rotation.py"""
    grid = defaultdict(list)
    for idx, p in enumerate(points):
        x, y = p[0], p[1]
        cx = int(np.floor(x / cell_size))
        cy = int(np.floor(y / cell_size))
        grid[(cx, cy)].append(idx)
    return grid


def nearest_neighbor_map(source_frame, target_frame):
    """Tìm điểm đích tối ưu nhất cho từng điểm nguồn từ rotation.py"""
    grid = build_spatial_hash(target_frame, CELL_SIZE)
    thr2 = MAX_SEARCH_RADIUS * MAX_SEARCH_RADIUS
    r_cells = int(np.ceil(MAX_SEARCH_RADIUS / CELL_SIZE))

    best_target_idx = np.full(len(source_frame), -1, dtype=np.int64)

    for src_idx, p in enumerate(source_frame):
        sx, sy, sint = p[0], p[1], p[2]
        cx = int(np.floor(sx / CELL_SIZE))
        cy = int(np.floor(sy / CELL_SIZE))

        candidate_ids = []
        for dx_cell in range(-r_cells, r_cells + 1):
            for dy_cell in range(-r_cells, r_cells + 1):
                candidate_ids.extend(grid.get((cx + dx_cell, cy + dy_cell), []))

        if not candidate_ids:
            continue

        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        cand = target_frame[candidate_ids]

        # Intensity filter
        int_diff = np.abs(cand[:, 2] - sint)
        mask_int = int_diff <= INTENSITY_TOLERANCE
        if not np.any(mask_int):
            continue

        candidate_ids = candidate_ids[mask_int]
        cand = cand[mask_int]

        # Distance filter
        dx = cand[:, 0] - sx
        dy = cand[:, 1] - sy
        d2 = dx * dx + dy * dy

        mask_dist = d2 <= thr2
        if not np.any(mask_dist):
            continue

        candidate_ids = candidate_ids[mask_dist]
        d2 = d2[mask_dist]

        if d2.size == 1:
            best_target_idx[src_idx] = candidate_ids[0]
            continue

        # Ratio test: loại bỏ các liên kết mập mờ
        top2 = np.argpartition(d2, 1)[:2]
        top2 = top2[np.argsort(d2[top2])]
        best_i, second_i = top2[0], top2[1]

        if d2[best_i] / (d2[second_i] + 1e-12) > NN_RATIO_TEST:
            continue

        best_target_idx[src_idx] = candidate_ids[best_i]

    return best_target_idx


def match_spatial_hash_mutual(old_frame, new_frame):
    """Khớp cặp hai chiều (Mutual Nearest Neighbor) từ rotation.py"""
    old_to_new = nearest_neighbor_map(old_frame, new_frame)
    new_to_old = nearest_neighbor_map(new_frame, old_frame)

    matches = []
    for old_idx, new_idx in enumerate(old_to_new):
        if new_idx == -1:
            continue
        if new_to_old[new_idx] == old_idx:
            matches.append((old_frame[old_idx], new_frame[new_idx]))
    return matches


def solve_two_point(p1, p2, q1, q2):
    """Tính toán biến đổi hình học từ 2 cặp điểm tương ứng"""
    p1xy, p2xy = p1[:2], p2[:2]
    q1xy, q2xy = q1[:2], q2[:2]

    vp = p2xy - p1xy
    vq = q2xy - q1xy

    if np.hypot(vp[0], vp[1]) < 1e-12 or np.hypot(vq[0], vq[1]) < 1e-12:
        return None

    theta = np.arctan2(vq[1], vq[0]) - np.arctan2(vp[1], vp[0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    t1 = q1xy - (R @ p1xy)
    t2 = q2xy - (R @ p2xy)
    t = 0.5 * (t1 + t2)

    return R, float(t[0]), float(t[1])


def estimate_rigid_transform(inliers):
    """Khớp bình phương tối thiểu tinh chỉnh ma trận biến đổi (Kabsch SVD 2D) từ rotation.py"""
    if len(inliers) == 0:
        return np.eye(2, dtype=np.float64), 0.0, 0.0

    P = np.array([m[0][:2] for m in inliers], dtype=np.float64)
    Q = np.array([m[1][:2] for m in inliers], dtype=np.float64)

    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)

    Pc = P - mu_P
    Qc = Q - mu_Q

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_Q - R @ mu_P
    return R, float(t[0]), float(t[1])


def ransac_rigid_transform(matches):
    """Ước lượng biến đổi cứng bằng RANSAC từ rotation.py"""
    n = len(matches)
    if n < 2:
        return np.eye(2, dtype=np.float64), 0.0, 0.0, [], []

    P = np.array([m[0][:2] for m in matches], dtype=np.float64)
    Q = np.array([m[1][:2] for m in matches], dtype=np.float64)

    rng = np.random.default_rng(42)
    best_inlier_count = 0
    best_score = np.inf
    best_mask = np.zeros(n, dtype=bool)
    thresh_sq = INLIER_THRESHOLD * INLIER_THRESHOLD

    for _ in range(RANSAC_ITERATIONS):
        idx1, idx2 = rng.choice(n, size=2, replace=False)

        sol = solve_two_point(
            matches[idx1][0], matches[idx2][0],
            matches[idx1][1], matches[idx2][1],
        )
        if sol is None:
            continue

        R, tx, ty = sol
        t = np.array([tx, ty], dtype=np.float64)

        Q_pred = (R @ P.T).T + t
        err2 = np.sum((Q - Q_pred) ** 2, axis=1)

        mask = err2 < thresh_sq
        count = int(mask.sum())
        if count == 0:
            continue

        score = float(err2[mask].sum())

        if count > best_inlier_count or (count == best_inlier_count and score < best_score):
            best_inlier_count = count
            best_score = score
            best_mask = mask.copy()

            if count / n >= EARLY_STOP_INLIER_RATIO:
                break

    inlier_matches = [matches[i] for i in np.where(best_mask)[0]]
    outlier_matches = [matches[i] for i in np.where(~best_mask)[0]]

    R_refined, tx_refined, ty_refined = estimate_rigid_transform(inlier_matches)
    return R_refined, tx_refined, ty_refined, inlier_matches, outlier_matches


# ============================================================
# 2. MÔ-ĐUN GIẢI ATTITUDE 3D & TÍNH SAI SỐ
# ============================================================
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Chuyển ma trận quay 3x3 sang Quaternion q = [qw, qx, qy, qz]"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])
    return q / np.linalg.norm(q)


def calculate_quaternion_error_arcsec(q_true: np.ndarray, q_est: np.ndarray) -> float:
    """Tính sai số định hướng giữa 2 Quaternion theo đơn vị Arcsecond"""
    dot_product = np.abs(np.dot(q_true, q_est))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot_product)
    return float(np.degrees(angle_rad) * 3600.0)


def solve_attitude_wahba_svd(inlier_matches, catalog_dict, focal_length_px, cx, cy):
    """Giải bài toán Wahba khôi phục Quaternion 3D R_ECI->CAM từ danh sách Inlier Matches"""
    W, V = [], []

    for _, q in inlier_matches:
        star_id = int(q[3])
        # Chỉ xét các sao thật có trong Catalog (star_id >= 0)
        if star_id >= 0 and star_id in catalog_dict:
            # Vectơ hướng đơn vị trong hệ tọa độ Camera 3D
            x_c = (q[0] - cx) / focal_length_px
            y_c = (q[1] - cy) / focal_length_px
            v_cam = np.array([x_c, y_c, 1.0], dtype=np.float64)
            v_cam /= np.linalg.norm(v_cam)

            # Vectơ tham chiếu 3D ECI từ Catalog
            cat_star = catalog_dict[star_id]
            v_eci = np.array([cat_star['v_eci_x'], cat_star['v_eci_y'], cat_star['v_eci_z']], dtype=np.float64)

            W.append(v_cam)
            V.append(v_eci)

    if len(W) < 3:
        return None

    W = np.array(W)
    V = np.array(V)

    # Giải bài toán Wahba bằng SVD
    B = W.T @ V
    U, _, Vt = np.linalg.svd(B)
    M = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        M[2, 2] = -1.0

    R_est = U @ M @ Vt
    return rotation_matrix_to_quaternion(R_est)


# ============================================================
# 3. CHƯƠNG TRÌNH CHẠY THỬ NGHIỆM ĐÁNH GIÁ DATASET
# ============================================================
def run_dataset_tracking_pipeline(dataset_dir: str = "./star_tracker_dataset"):
    print("\n" + "=" * 80)
    print(" CHẠY THỬ NGHIỆM THUẬT TOÁN ROTATION.PY TRÊN DATASET TRACKING ".center(80, "="))
    print("=" * 80)

    # Nạp dữ liệu
    catalog_df = pd.read_csv(os.path.join(dataset_dir, "star_catalog.csv"))
    obs_df = pd.read_csv(os.path.join(dataset_dir, "tracking_observations.csv"))
    gt_df = pd.read_csv(os.path.join(dataset_dir, "ground_truth_attitude.csv"))

    with open(os.path.join(dataset_dir, "dataset_metadata.json"), 'r') as f:
        meta = json.load(f)

    sensor_meta = meta.get('sensor_specs', meta.get('sensor', {}))
    img_w = sensor_meta.get('image_width', sensor_meta.get('width', 500.0))
    img_h = sensor_meta.get('image_height', sensor_meta.get('height', 500.0))
    fov_deg = sensor_meta.get('fov_deg', 17.0)
    
    focal_length = sensor_meta.get('focal_length_px', (img_w / 2.0) / np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = img_w / 2.0, img_h / 2.0

    # Chuyển Catalog thành Dictionary tra cứu theo star_id
    catalog_dict = catalog_df.set_index('star_id').to_dict('index')

    num_frames = len(gt_df)
    eval_results = []

    print(f"\n>>> Đang xử lý chuỗi {num_frames} khung hình...")

    for f_idx in range(1, num_frames):
        # Lấy dữ liệu quan sát dạng mảng [x, y, intensity, star_id] chuẩn của rotation.py
        prev_obs = obs_df[obs_df['frame_id'] == f_idx - 1][['x_measured', 'y_measured', 'intensity_measured', 'star_id']].values
        curr_obs = obs_df[obs_df['frame_id'] == f_idx][['x_measured', 'y_measured', 'intensity_measured', 'star_id']].values

        t0 = time.perf_counter()

        # 1. Khớp cặp bằng Spatial Hash Mutual Nearest Neighbor (từ rotation.py)
        matches = match_spatial_hash_mutual(prev_obs, curr_obs)

        # 2. Lọc nhiễu hình học 2D bằng RANSAC (từ rotation.py)
        R_2d, tx_2d, ty_2d, inlier_matches, outlier_matches = ransac_rigid_transform(matches)

        # 3. Tính toán Attitude 3D từ danh sách Inliers
        q_est = solve_attitude_wahba_svd(inlier_matches, catalog_dict, focal_length, cx, cy)

        t_proc = (time.perf_counter() - t0) * 1000.0 # ms

        # 4. Lấy Ground Truth Quaternion
        gt_row = gt_df[gt_df['frame_id'] == f_idx].iloc[0]
        q_gt = np.array([gt_row['q_w'], gt_row['q_x'], gt_row['q_y'], gt_row['q_z']])

        # 5. Tính sai số Attitude
        att_err_arcsec = calculate_quaternion_error_arcsec(q_gt, q_est) if q_est is not None else np.nan

        # Đánh giá Precision của bước Khớp cặp
        correct_matches = sum(1 for p, q in inlier_matches if int(p[3]) == int(q[3]) and int(p[3]) >= 0)
        precision = (100.0 * correct_matches / len(inlier_matches)) if len(inlier_matches) > 0 else 0.0

        eval_results.append({
            'frame_id': f_idx,
            'proc_time_ms': t_proc,
            'total_matches': len(matches),
            'inliers_count': len(inlier_matches),
            'precision_pct': precision,
            'att_err_arcsec': att_err_arcsec
        })

    res_df = pd.DataFrame(eval_results)

    # In báo cáo tổng hợp
    print("\n" + "=" * 80)
    print(" BÁO CÁO KẾT QUẢ HIỆU NĂNG THUẬT TOÁN ROTATION.PY ".center(80, "-"))
    print("=" * 80)
    print(f" * Thời gian xử lý trung bình / Frame : {res_df['proc_time_ms'].mean():.2f} ms")
    print(f" * Số liên kết Inliers trung bình/Frame : {res_df['inliers_count'].mean():.1f}")
    print(f" * Độ chính xác khớp cặp (Precision)  : {res_df['precision_pct'].mean():.2f} %")
    print(f" * Sai số định hướng RMS (Attitude Error): {res_df['att_err_arcsec'].mean():.2f} arcseconds")
    print("=" * 80)

    # Trực quan hóa kết quả
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(res_df['frame_id'], res_df['att_err_arcsec'], 'r-o', markersize=3, label="Attitude Error (Arcsec)")
    axes[0].axhline(y=res_df['att_err_arcsec'].mean(), color='darkred', linestyle='--', label=f"RMS Mean ({res_df['att_err_arcsec'].mean():.2f}'')")
    axes[0].set_title("Sai số Định hướng Attitude 3D (Arcsec)")
    axes[0].set_xlabel("Frame ID")
    axes[0].set_ylabel("Error (arcsecond)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(res_df['frame_id'], res_df['proc_time_ms'], 'b-s', markersize=3, label="Processing Time")
    axes[1].axhline(y=100.0, color='g', linestyle='--', label='10 Hz Real-time limit (100ms)')
    axes[1].set_title("Thời gian Thực thi Thuật toán (ms)")
    axes[1].set_xlabel("Frame ID")
    axes[1].set_ylabel("Time (ms)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(dataset_dir, "rotation_algorithm_performance.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\n>>> Đã xuất đồ thị báo cáo hiệu năng: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Đảm bảo bạn đã chạy `python create_dataset.py` trước khi chạy script này
    run_dataset_tracking_pipeline(dataset_dir="./star_tracker_dataset")