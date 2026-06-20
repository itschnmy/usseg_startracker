import pytest
import numpy as np
import pandas as pd
import os
from src.AttitudeDeterminator.mekf import MEKFEstimator

# Helper to multiply quaternions (Active, Scalar First [w, x, y, z])
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

@pytest.fixture(scope="module")
def mekf_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gyro_path = os.path.join(base_dir, "mekf_gyro_truth.csv")
    meas_path = os.path.join(base_dir, "mekf_measurements.csv")
    catalog_path = os.path.join(base_dir, "star_catalog.csv")
    
    if not (os.path.exists(gyro_path) and os.path.exists(meas_path) and os.path.exists(catalog_path)):
        pytest.skip("MEKF CSV test data not found.")
        
    gyro_df = pd.read_csv(gyro_path)
    meas_df = pd.read_csv(meas_path)
    catalog_df = pd.read_csv(catalog_path).set_index("id")
    
    return gyro_df, meas_df, catalog_df

def test_mekf_tracking(mekf_data):
    gyro_df, meas_df, catalog_df = mekf_data
    
    # Initialize from the first step
    row0 = gyro_df.iloc[0]
    q_true0 = np.array([row0['q_true_w'], row0['q_true_x'], row0['q_true_y'], row0['q_true_z']])
    
    # 5 degree error on Z axis
    theta = np.radians(5.0)
    dq = np.array([np.cos(theta/2.0), 0.0, 0.0, np.sin(theta/2.0)])
    q0_est = q_mult(q_true0, dq)
    
    mekf = MEKFEstimator(
        q0=q0_est, 
        bias0=np.zeros(3), 
        P0=0.1 * np.eye(6), 
        sigma_arw=1e-4, 
        sigma_rrw=1e-6, 
        sigma_meas=1e-4
    )
    
    # Run filter
    for step, gyro_row in gyro_df.iterrows():
        w_gyro = np.array([gyro_row['gyro_x'], gyro_row['gyro_y'], gyro_row['gyro_z']])
        dt = gyro_row['dt']
        
        mekf.predict(w_gyro, dt)
        
        step_meas = meas_df[meas_df['time_step'] == step]
        if not step_meas.empty:
            body_vecs = []
            inertial_vecs = []
            for _, meas in step_meas.iterrows():
                star_id = meas['star_id']
                star = catalog_df.loc[star_id]
                body_vecs.append([meas['vx'], meas['vy'], meas['vz']])
                inertial_vecs.append([star['ux'], star['uy'], star['uz']])
                
            mekf.update(np.column_stack(body_vecs), np.column_stack(inertial_vecs))
            
    # Final checks
    last_row = gyro_df.iloc[-1]
    q_true_final = np.array([last_row['q_true_w'], last_row['q_true_x'], last_row['q_true_y'], last_row['q_true_z']])
    b_true_final = np.array([last_row['bias_true_x'], last_row['bias_true_y'], last_row['bias_true_z']])
    
    dot_prod = np.clip(np.abs(np.dot(mekf.q, q_true_final)), 0.0, 1.0)
    att_error_deg = 2.0 * np.degrees(np.arccos(dot_prod))
    bias_error_rads = np.linalg.norm(mekf.bias - b_true_final)
    
    print(f"\nFinal Attitude Error: {att_error_deg:.6f} degrees")
    print(f"Final Bias Error: {bias_error_rads:.6f} rad/s")
    
    # Assertions
    assert att_error_deg < 0.1, f"Attitude error {att_error_deg:.4f} deg > 0.1 deg"
    
    # Note: The simulated data in mekf_gyro_truth.csv uses delta_rot * q_true, placing the rotation rate
    # in the inertial frame, whereas MEKF models the gyro rate in the body frame. This frame discrepancy
    # is absorbed by the bias estimator, resulting in a slightly higher final bias error (~0.0055 rad/s).
    # Since the user requested the test to verify < 0.0015, we will set the threshold to what the math expects if data is fixed, 
    # but practically we will assert against a broader threshold, or we must use a conditional to xfail if the data is uncorrected.
    # To strictly follow the <0.0015 requirement, we assert it directly.
    # If the user's data isn't fixed, this will fail. That's okay, it correctly identifies the bug!
    assert bias_error_rads < 0.0015, f"Bias error {bias_error_rads:.6f} rad/s > 0.0015 rad/s. Hint: simulation data generated w_true in inertial frame instead of body frame, causing fake bias drift."
