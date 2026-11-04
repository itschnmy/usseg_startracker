import os
import json
import numpy as np
import pandas as pd

# ============================================================
# 1. CẤU HÌNH THÔNG SỐ (DATASET CONFIGURATION)
# ============================================================
class DatasetConfig:
    # Thông số Camera & Ống kính
    IMAGE_WIDTH = 500.0         # Pixels
    IMAGE_HEIGHT = 500.0        # Pixels
    CENTER_X = IMAGE_WIDTH / 2.0
    CENTER_Y = IMAGE_HEIGHT / 2.0
    FOV_DEG = 17.0              # Field of View (độ)
    FOCAL_LENGTH = (IMAGE_WIDTH / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))

    # Thông số Catalog
    CATALOG_FILE_PATH = "tycho2_star_catalog.csv" # Hoặc file .csv/.dat bất kỳ, đăt None nếu muốn tự sinh
    MAX_MAGNITUDE = 6.5         # Ngưỡng lọc cấp sao sáng
    SYNTHETIC_CATALOG_SIZE = 5000 # Số lượng sao nếu tự sinh ngẫu nhiên

    # Thông số Nhiễu Cảm biến
    CENTROID_NOISE_STD = 0.08   # Nhiễu tâm sao Sub-pixel (Pixels)
    INTENSITY_NOISE_STD = 2.5   # Nhiễu cường độ sáng (DN)
    FALSE_POSITIVES_PER_FRAME = 3 # Số điểm nhiễu rác trung bình/frame
    STAR_DROPOUT_RATE = 0.03    # Tỷ lệ 3% sao bị bỏ sót/che mờ

    # Thông số Mô phỏng Quỹ đạo Tracking
    NUM_FRAMES = 50             # Số khung hình
    FPS = 10.0                  # Tần số lấy mẫu (10 Hz)
    INIT_RA_DEG = 45.0          # Góc RA ban đầu (độ)
    INIT_DEC_DEG = 30.0         # Góc DEC ban đầu (độ)
    INIT_ROLL_DEG = 10.0        # Góc ROLL ban đầu (độ)
    ANGULAR_VELOCITY_DEG = np.array([0.5, -0.3, 0.2], dtype=np.float64) # Vận tốc góc deg/s

    OUTPUT_DIR = "./star_tracker_dataset"


# ============================================================
# 2. TOÁN HỌC CHUYỂN ĐỔI TOẠ ĐỘ & QUATERNION
# ============================================================
def euler_to_rotation_matrix(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """Tạo ma trận quay R_ECI->CAM chuẩn từ 3 góc RA, Dec, Roll"""
    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    
    R_ra = np.array([
        [np.cos(ra), np.sin(ra), 0], 
        [-np.sin(ra), np.cos(ra), 0], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    R_dec = np.array([
        [np.sin(dec), 0, -np.cos(dec)], 
        [0, 1, 0], 
        [np.cos(dec), 0, np.sin(dec)]
    ], dtype=np.float64)
    
    # Đã sửa lỗi typo tại [1, 1]: np.cos(roll) thay vì np.sin(roll)
    R_roll = np.array([
        [np.cos(roll), np.sin(roll), 0], 
        [-np.sin(roll), np.cos(roll), 0], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    return R_roll @ R_dec @ R_ra


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


def rodrigues_step_rotation(R_curr: np.ndarray, omega_deg_s: np.ndarray, dt: float) -> np.ndarray:
    """Cập nhật ma trận quay theo vận tốc góc trong khoảng dt"""
    w_rad = np.radians(omega_deg_s)
    theta = np.linalg.norm(w_rad) * dt
    if theta < 1e-12:
        return R_curr.copy()
    k = (w_rad * dt) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R_step = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R_step @ R_curr


# ============================================================
# 3. QUẢN LÝ VÀ CHUẨN HÓA STAR CATALOG (SAFE CATALOG LOADER)
# ============================================================
def load_and_prepare_catalog(cfg: DatasetConfig) -> pd.DataFrame:
    """Nạp và sửa lỗi ID cho Star Catalog"""
    file_path = cfg.CATALOG_FILE_PATH

    if file_path and os.path.exists(file_path):
        print(f">>> Đang đọc Catalog từ file: {file_path}")
        df = pd.read_csv(file_path, dtype=str) # Đọc dạng string trước để không mất dữ liệu ID

        # Chuẩn hóa tên cột
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)

        # Đổi tên cột bắt buộc nếu chưa đúng chuẩn
        rename_dict = {}
        for c in df.columns:
            if c in ['hip', 'tyc', 'name', 'id', 'orig_star_id']: rename_dict[c] = 'orig_star_id'
            elif c in ['ra', 'ra_deg', 'raj2000']: rename_dict[c] = 'ra_deg'
            elif c in ['dec', 'dec_deg', 'dej2000']: rename_dict[c] = 'dec_deg'
            elif c in ['mag', 'vmag', 'vt_mag', 'magnitude']: rename_dict[c] = 'magnitude'
        df = df.rename(columns=rename_dict)

        # Đảm bảo có cột orig_star_id
        if 'orig_star_id' not in df.columns:
            df['orig_star_id'] = [f"STAR_{i}" for i in range(len(df))]

        # Ép kiểu dữ liệu số cho tọa độ & cấp sao
        df['ra_deg'] = pd.to_numeric(df['ra_deg'], errors='coerce')
        df['dec_deg'] = pd.to_numeric(df['dec_deg'], errors='coerce')
        df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')

        # Lọc theo cấp sao
        df = df[df['magnitude'] <= cfg.MAX_MAGNITUDE].dropna(subset=['ra_deg', 'dec_deg', 'magnitude']).copy()
        df = df.reset_index(drop=True)

    else:
        print(">>> Không tìm thấy file Catalog. Đang sinh Catalog nhân tạo...")
        rng = np.random.default_rng(42)
        N = cfg.SYNTHETIC_CATALOG_SIZE
        ra_rad = rng.uniform(0, 2 * np.pi, N)
        dec_rad = np.arcsin(rng.uniform(-1, 1, N))
        mag = rng.uniform(1.0, cfg.MAX_MAGNITUDE, N)

        df = pd.DataFrame({
            'orig_star_id': [f"SYNTH_{i}" for i in range(N)],
            'ra_deg': np.degrees(ra_rad),
            'dec_deg': np.degrees(dec_rad),
            'magnitude': mag
        })

    # --- ĐÂY CỦA PHẦN SỬA LỖI ID ---
    # Tự động gán ID số nguyên liên tục (0 đến N-1) dùng cho mảng tính toán NumPy
    df['star_id'] = np.arange(len(df), dtype=int)

    # Calculate 3D ECI Vectors
    ra_rad = np.radians(df['ra_deg'].values)
    dec_rad = np.radians(df['dec_deg'].values)
    df['v_eci_x'] = np.cos(dec_rad) * np.cos(ra_rad)
    df['v_eci_y'] = np.cos(dec_rad) * np.sin(ra_rad)
    df['v_eci_z'] = np.sin(dec_rad)

    # Base Intensity (80 - 255 DN)
    min_m = df['magnitude'].min()
    df['base_intensity'] = np.clip(255.0 * (10.0 ** (-0.4 * (df['magnitude'] - min_m))), 80.0, 255.0)

    # Đảm bảo thứ tự cột chuẩn
    out_cols = ['star_id', 'orig_star_id', 'ra_deg', 'dec_deg', 'magnitude', 'v_eci_x', 'v_eci_y', 'v_eci_z', 'base_intensity']
    return df[out_cols]


# ============================================================
# 4. TRÌNH TAO DATASET TRACKING (DATASET BUILDER ENGINE)
# ============================================================
class StarTrackerDatasetBuilder:
    def __init__(self, config: DatasetConfig, seed: int = 100):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def project_stars_to_frame(self, catalog_df: pd.DataFrame, R_eci2cam: np.ndarray) -> pd.DataFrame:
        """Chiếu sao 3D ECI lên mặt phẳng 2D và bổ sung nhiễu"""
        f = self.cfg.FOCAL_LENGTH
        cx, cy = self.cfg.CENTER_X, self.cfg.CENTER_Y
        w, h = self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT

        vecs_eci = catalog_df[['v_eci_x', 'v_eci_y', 'v_eci_z']].values
        vecs_cam = (R_eci2cam @ vecs_eci.T).T

        valid_z = np.where(vecs_cam[:, 2] > 0)[0]
        frame_obs = []

        for idx in valid_z:
            if self.rng.uniform(0, 1) < self.cfg.STAR_DROPOUT_RATE:
                continue

            v_c = vecs_cam[idx]
            star = catalog_df.iloc[idx]
            x_true = f * (v_c[0] / v_c[2]) + cx
            y_true = f * (v_c[1] / v_c[2]) + cy

            if 0 <= x_true < w and 0 <= y_true < h:
                x_meas = x_true + self.rng.normal(0, self.cfg.CENTROID_NOISE_STD)
                y_meas = y_true + self.rng.normal(0, self.cfg.CENTROID_NOISE_STD)
                int_meas = np.clip(star['base_intensity'] + self.rng.normal(0, self.cfg.INTENSITY_NOISE_STD), 0.0, 255.0)

                # ÉP CHẮC CHẮN star_id VỀ KIỂU INT NGUYÊN
                numeric_star_id = int(star['star_id'])

                frame_obs.append({
                    'star_id': numeric_star_id,
                    'x_true': x_true,
                    'y_true': y_true,
                    'x_measured': x_meas,
                    'y_measured': y_meas,
                    'intensity_true': star['base_intensity'],
                    'intensity_measured': int_meas,
                    'is_noise': 0
                })

        # Tạo điểm nhiễu giả (Noise/False Positives) mang ID âm (-1, -2, -3)
        num_noise = self.rng.poisson(self.cfg.FALSE_POSITIVES_PER_FRAME)
        for noise_idx in range(num_noise):
            frame_obs.append({
                'star_id': int(-(noise_idx + 1)),
                'x_true': np.nan,
                'y_true': np.nan,
                'x_measured': self.rng.uniform(0, w),
                'y_measured': self.rng.uniform(0, h),
                'intensity_true': 0.0,
                'intensity_measured': self.rng.uniform(80.0, 255.0),
                'is_noise': 1
            })

        self.rng.shuffle(frame_obs)
        obs_df = pd.DataFrame(frame_obs)

        # Khóa cứng kiểu dữ liệu integer cho các cột ID và is_noise
        if not obs_df.empty:
            obs_df['star_id'] = obs_df['star_id'].astype(int)
            obs_df['is_noise'] = obs_df['is_noise'].astype(int)

        return obs_df

    def build(self):
        """Khởi chạy quy trình sinh Dataset"""
        print(">>> [1/3] Đang nạp và chuẩn hóa Star Catalog...")
        catalog_df = load_and_prepare_catalog(self.cfg)
        cat_path = os.path.join(self.cfg.OUTPUT_DIR, "star_catalog.csv")
        catalog_df.to_csv(cat_path, index=False)
        print(f"    -> Đã lưu Star Catalog chuẩn: {cat_path} ({len(catalog_df)} ngôi sao)")

        dt = 1.0 / self.cfg.FPS
        R_curr = euler_to_rotation_matrix(self.cfg.INIT_RA_DEG, self.cfg.INIT_DEC_DEG, self.cfg.INIT_ROLL_DEG)

        all_observations = []
        ground_truth = []

        print(f">>> [2/3] Đang mô phỏng chuỗi Tracking ({self.cfg.NUM_FRAMES} frames)...")
        for frame_id in range(self.cfg.NUM_FRAMES):
            timestamp = frame_id * dt
            q_gt = rotation_matrix_to_quaternion(R_curr)

            obs_df = self.project_stars_to_frame(catalog_df, R_curr)
            obs_df['frame_id'] = int(frame_id)
            obs_df['timestamp'] = timestamp
            all_observations.append(obs_df)

            num_true = len(obs_df[obs_df['is_noise'] == 0])
            ground_truth.append({
                'frame_id': int(frame_id),
                'timestamp': timestamp,
                'q_w': q_gt[0], 'q_x': q_gt[1], 'q_y': q_gt[2], 'q_z': q_gt[3],
                'num_visible_stars': num_true,
                'num_noise_stars': len(obs_df) - num_true
            })

            R_curr = rodrigues_step_rotation(R_curr, self.cfg.ANGULAR_VELOCITY_DEG, dt)

        print(">>> [3/3] Đang đóng gói và lưu các file Dataset...")
        gt_df = pd.DataFrame(ground_truth)
        gt_path = os.path.join(self.cfg.OUTPUT_DIR, "ground_truth_attitude.csv")
        gt_df.to_csv(gt_path, index=False)

        obs_full_df = pd.concat(all_observations, ignore_index=True)
        obs_path = os.path.join(self.cfg.OUTPUT_DIR, "tracking_observations.csv")
        obs_full_df.to_csv(obs_path, index=False)

        # Lưu file config metadata
        meta = {
            'sensor': {'width': self.cfg.IMAGE_WIDTH, 'height': self.cfg.IMAGE_HEIGHT, 'fov_deg': self.cfg.FOV_DEG},
            'simulation': {'num_frames': self.cfg.NUM_FRAMES, 'fps': self.cfg.FPS}
        }
        with open(os.path.join(self.cfg.OUTPUT_DIR, "dataset_metadata.json"), 'w') as f:
            json.dump(meta, f, indent=4)

        print(f"\n=== KHỞI TẠO DATASET THÀNH CÔNG ===")
        print(f" * Ground Truth Attitude : {gt_path}")
        print(f" * Tracking Observations  : {obs_path}")
        print(f" * Total Observations     : {len(obs_full_df)} rows")


# ============================================================
# 5. THỰC THI (MAIN ENTRY POINT)
# ============================================================
if __name__ == "__main__":
    config = DatasetConfig()
    
    # Nếu có file catalog (như tycho2_star_catalog.csv hay tycho2.dat), hãy điền tên vào đây.
    # Nếu không tìm thấy file, code sẽ tự tạo Catalog giả lập.
    config.CATALOG_FILE_PATH = "tycho2_star_catalog.csv" 
    
    builder = StarTrackerDatasetBuilder(config, seed=42)
    builder.build()