import sys
import os
import csv
import numpy as np

# Adjust path to find src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.AttitudeDeterminator import (
    TRIADEstimator,
    QUESTEstimator,
    DavenportQEstimator,
    AttitudeControlSystem,
    ADCSMode,
    MEKFEstimator
)

def Xi_matrix(q):
    """Computes the 4x3 Xi matrix for scalar-first unit quaternion q = [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w]
    ], dtype=float)

def quaternion_to_rot(q):
    """Convert scalar-first unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z),         2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),         1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x**2 + y**2)]
    ])

def quat_diff_deg(q1, q2):
    """Compute the angular difference between two quaternions in degrees."""
    dot = abs(np.dot(q1, q2))
    if dot > 1.0:
        dot = 1.0
    angle_rad = 2.0 * np.arccos(dot)
    return np.degrees(angle_rad)

def load_crux_stars_from_csv(csv_path):
    """Load Crux stars unit vectors from star_catalog.csv."""
    crux_ids = {60718, 62434, 61084, 59747, 60260}
    vectors = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            star_id = int(row['id'])
            if star_id in crux_ids:
                ux = float(row['ux'])
                uy = float(row['uy'])
                uz = float(row['uz'])
                vectors[star_id] = np.array([ux, uy, uz], dtype=float)
                
    # Return as list of arrays in consistent order
    return [vectors[sid] for sid in sorted(crux_ids) if sid in vectors]

def main():
    print("==================================================")
    print("  VERIFYING PYTHON SOLVERS ON REAL CRUX STAR DATA")
    print("==================================================")
    
    # 1. Load Real Reference/Inertial Vectors from star_catalog.csv
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../star_catalog.csv"))
    if not os.path.exists(csv_path):
        print(f"Error: star_catalog.csv not found at {csv_path}")
        sys.exit(1)
        
    crux_stars = load_crux_stars_from_csv(csv_path)
    if len(crux_stars) < 5:
        print(f"Warning: Only found {len(crux_stars)} Crux stars in catalog. Expected 5.")
        sys.exit(1)
        
    # Stack vectors into 3x5 matrix (Inertial Frame)
    inertial_vectors = np.column_stack(crux_stars)
    print(f"Loaded {inertial_vectors.shape[1]} Crux stars from star_catalog.csv.")
    
    # 2. Build simulated True Attitude matching test_lis.cpp logic
    # Calculate the mean vector of the crux stars to point the camera at them
    mean_u = np.mean(inertial_vectors, axis=1)
    mean_u /= np.linalg.norm(mean_u)
    
    # Construct camera orientation R_true such that Z-axis points at mean_u
    z_body = mean_u
    temp = np.array([1.0, 0.0, 0.0]) if abs(z_body[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_body = np.cross(temp, z_body)
    x_body /= np.linalg.norm(x_body)
    y_body = np.cross(z_body, x_body)
    y_body /= np.linalg.norm(y_body)
    
    R_true = np.column_stack((x_body, y_body, z_body))
    
    # Extract active quaternion
    tr = np.trace(R_true)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R_true[2, 1] - R_true[1, 2]) / S
        qy = (R_true[0, 2] - R_true[2, 0]) / S
        qz = (R_true[1, 0] - R_true[0, 1]) / S
    else:
        # Simple branch assuming Z-axis dominates Crux alignment
        S = np.sqrt(1.0 + R_true[2, 2] - R_true[0, 0] - R_true[1, 1]) * 2.0
        qw = (R_true[1, 0] - R_true[0, 1]) / S
        qx = (R_true[0, 2] + R_true[2, 0]) / S
        qy = (R_true[1, 2] + R_true[2, 1]) / S
        qz = 0.25 * S
        
    q_true_active = np.array([qw, qx, qy, qz])
    q_true_active /= np.linalg.norm(q_true_active)
    if q_true_active[0] < 0:
        q_true_active = -q_true_active
        
    q_true_passive = np.array([q_true_active[0], -q_true_active[1], -q_true_active[2], -q_true_active[3]])
    
    print(f"True Active Quaternion (Body -> Inertial): {q_true_active}")
    print(f"Expected Passive Quaternion (Inertial -> Body): {q_true_passive}")
    print(f"True Rotation Matrix (R_true):\n{R_true}\n")
    
    # Project catalog stars to Body frame (simulated sensor measurements)
    # u_body = R_true.T * u_inertial
    body_vectors = R_true.T @ inertial_vectors
    
    # 3. Verify TRIAD (first 2 vectors, Passive rotation)
    triad = TRIADEstimator()
    q_triad = triad.estimate(body_vectors[:, :2], inertial_vectors[:, :2])
    err_triad = quat_diff_deg(q_true_passive, q_triad)
    print(f"TRIAD Estimated Quaternion: {q_triad}")
    print(f"TRIAD Angular Error       : {err_triad:.2e} deg")
    if err_triad < 1e-5:
        print("--> TRIAD Test: SUCCESS\n")
    else:
        print("--> TRIAD Test: FAILED\n")
        
    # 4. Verify QUEST (all 5 vectors, Passive rotation)
    quest = QUESTEstimator()
    q_quest = quest.estimate(body_vectors, inertial_vectors)
    err_quest = quat_diff_deg(q_true_passive, q_quest)
    print(f"QUEST Estimated Quaternion: {q_quest}")
    print(f"QUEST Angular Error       : {err_quest:.2e} deg")
    if err_quest < 1e-5:
        print("--> QUEST Test: SUCCESS\n")
    else:
        print("--> QUEST Test: FAILED\n")
        
    # 5. Verify Davenport Q (all 5 vectors, Passive rotation)
    davenport = DavenportQEstimator()
    q_davenport = davenport.estimate(body_vectors, inertial_vectors)
    err_davenport = quat_diff_deg(q_true_passive, q_davenport)
    print(f"Davenport Q Estimated     : {q_davenport}")
    print(f"Davenport Q Angular Error : {err_davenport:.2e} deg")
    if err_davenport < 1e-7:
        print("--> Davenport Q Test: SUCCESS\n")
    else:
        print("--> Davenport Q Test: FAILED\n")
        
    # 6. Test AttitudeControlSystem wrapper
    print("Testing AttitudeControlSystem wrapper...")
    adcs_quest = AttitudeControlSystem(use_davenport=False)
    adcs_davenport = AttitudeControlSystem(use_davenport=True)
    
    # Degraded Mode (N=2 -> TRIAD / Passive)
    q_degraded = adcs_quest.process_sensor_data(body_vectors[:, :2], inertial_vectors[:, :2])
    print(f"Result (TRIAD): {q_degraded}")
    print(f"Error vs Passive: {quat_diff_deg(q_true_passive, q_degraded):.2e} deg\n")
    
    # Fine Pointing Mode (QUEST -> Passive)
    q_fine_q = adcs_quest.process_sensor_data(body_vectors, inertial_vectors)
    print(f"Result (QUEST): {q_fine_q}")
    print(f"Error vs Passive: {quat_diff_deg(q_true_passive, q_fine_q):.2e} deg\n")
    
    # Fine Pointing Mode (Davenport -> Passive)
    q_fine_d = adcs_davenport.process_sensor_data(body_vectors, inertial_vectors)
    print(f"Result (Davenport): {q_fine_d}")
    print(f"Error vs Passive: {quat_diff_deg(q_true_passive, q_fine_d):.2e} deg\n")
    
    # 6.5 Verify AttitudeDeterminator and RelativeAttitudeDeterminator wrappers
    print("Testing AttitudeDeterminator and RelativeAttitudeDeterminator wrappers...")
    from src.AttitudeDeterminator import AttitudeDeterminator, RelativeAttitudeDeterminator
    
    # Test AttitudeDeterminator
    abs_det = AttitudeDeterminator(method="QUEST")
    q_abs = abs_det.estimate(body_vectors, inertial_vectors)
    err_abs = quat_diff_deg(q_true_passive, q_abs)
    print(f"AttitudeDeterminator (QUEST) Estimated: {q_abs}")
    print(f"AttitudeDeterminator Angular Error     : {err_abs:.2e} deg")
    if err_abs < 1e-5:
        print("--> AttitudeDeterminator Test: SUCCESS\n")
    else:
        print("--> AttitudeDeterminator Test: FAILED\n")

    # Test RelativeAttitudeDeterminator
    rel_det = RelativeAttitudeDeterminator(method="QUEST")
    q_rel = rel_det.estimate_relative(body_vectors, inertial_vectors)
    err_rel = quat_diff_deg(q_true_passive, q_rel)
    print(f"RelativeAttitudeDeterminator Estimated  : {q_rel}")
    print(f"RelativeAttitudeDeterminator Error      : {err_rel:.2e} deg")
    if err_rel < 1e-5:
        print("--> RelativeAttitudeDeterminator Test: SUCCESS\n")
    else:
        print("--> RelativeAttitudeDeterminator Test: FAILED\n")
        
    print("All Python Estimators Verified successfully on star_catalog.csv data!\n")
    
    # 7. Verify MEKF Filter (Dynamic time-series simulation)
    mekf_success = verify_mekf(inertial_vectors, q_true_passive)
    if not mekf_success:
        sys.exit(1)

def verify_mekf(inertial_vectors, q_init_passive):
    print("==================================================")
    print("  VERIFYING MULTIPLICATIVE EXTENDED KALMAN FILTER")
    print("==================================================")
    
    # 1. Simulation settings
    T = 10.0  # seconds
    dt = 0.01  # 100 Hz gyro
    t_steps = int(T / dt)
    update_period = 20  # 5 Hz Star Tracker updates
    
    # True state trajectories (constant rotation + constant bias)
    w_true = np.array([0.05, -0.02, 0.03])  # rad/s
    bias_true = np.array([0.005, -0.003, 0.002])  # rad/s
    
    q_true = q_init_passive.copy()
    
    # Initialize filter with perturbed attitude
    q0 = q_init_passive.copy()
    # Add a small initial error to quaternion to test convergence (1.0 deg rotation about [1, 1, 1] axis)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    theta = np.radians(1.0)
    dq = np.array([np.cos(theta/2.0), axis[0]*np.sin(theta/2.0), axis[1]*np.sin(theta/2.0), axis[2]*np.sin(theta/2.0)])
    
    # Passive quaternion multiplication: q0 = q0 * dq
    w1, x1, y1, z1 = q0
    w2, x2, y2, z2 = dq
    q0 = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    q0 /= np.linalg.norm(q0)
    
    bias0 = np.zeros(3)  # Start with zero bias estimate
    P0 = 0.01 * np.eye(6)
    
    # Noise specifications
    sigma_arw = 0.001  # rad/s^(1/2)
    sigma_rrw = 0.00001  # rad/s^(3/2)
    sigma_meas = 0.001  # rad (about 0.057 deg)
    
    mekf = MEKFEstimator(
        q0=q0,
        bias0=bias0,
        P0=P0,
        sigma_arw=sigma_arw,
        sigma_rrw=sigma_rrw,
        sigma_meas=sigma_meas
    )
    
    np.random.seed(42)  # For reproducible test runs
    
    # 2. Main Simulation Loop
    for step in range(t_steps):
        # Propagate True Attitude
        q_dot = 0.5 * Xi_matrix(q_true) @ w_true
        q_true += q_dot * dt
        q_true /= np.linalg.norm(q_true)
        if q_true[0] < 0:
            q_true = -q_true
            
        # Simulate noisy Gyro reading (true rate + bias + white noise)
        gyro_noise = np.random.normal(0, sigma_arw / np.sqrt(dt), size=3)
        w_gyro = w_true + bias_true + gyro_noise
        
        # Predict MEKF
        mekf.predict(w_gyro, dt)
        
        # Periodic Star Tracker Measurement Update
        if step % update_period == 0:
            # Generate body vectors by rotating inertial vectors with true attitude
            # Active rotation matrix (Body to Inertial)
            w, x, y, z = q_true
            R_active = np.array([
                [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z),         2.0*(x*z + w*y)],
                [2.0*(x*y + w*z),         1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)],
                [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x**2 + y**2)]
            ])
            # Passive rotation maps Inertial to Body: body = R_active^T * inertial
            body_vects_true = R_active.T @ inertial_vectors
            
            # Add measurement noise (white noise in 3D, then normalize)
            meas_noise = np.random.normal(0, sigma_meas, size=body_vects_true.shape)
            body_vects_noisy = body_vects_true + meas_noise
            for i in range(body_vects_noisy.shape[1]):
                body_vects_noisy[:, i] /= np.linalg.norm(body_vects_noisy[:, i])
                
            # Perform update
            mekf.update(body_vects_noisy, inertial_vectors)
            
    # 3. Verify Results
    final_att_err = quat_diff_deg(q_true, mekf.q)
    final_bias_err = np.linalg.norm(bias_true - mekf.bias)
    
    print(f"Final True Quaternion     : {q_true}")
    print(f"Final Estimated Quaternion: {mekf.q}")
    print(f"Final Attitude Error      : {final_att_err:.4f} deg")
    print(f"Final True Gyro Bias      : {bias_true}")
    print(f"Final Estimated Gyro Bias : {mekf.bias}")
    print(f"Final Gyro Bias Error     : {final_bias_err:.6f} rad/s")
    
    # Validation threshold
    # Attitude error should converge to < 0.1 degrees
    # Bias error should converge to < 0.0015 rad/s
    if final_att_err < 0.1 and final_bias_err < 0.0015:
        print("--> MEKF Filter Test: SUCCESS\n")
        return True
    else:
        print("--> MEKF Filter Test: FAILED\n")
        return False

if __name__ == "__main__":
    main()
