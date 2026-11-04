import os
import pandas as pd
import numpy as np

def convert_tycho2_pipe_to_csv(
    dat_path: str, 
    output_csv: str = "tycho2_star_catalog.csv", 
    max_mag: float = 6.5
):
    """
    Chuyển đổi file Tycho-2 định dạng 32 cột phân cách bởi dấu '|' sang CSV cho Star Tracker.
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Không tìm thấy file: {dat_path}")

    print(f">>> Đang xử lý file Tycho-2: {dat_path}...")

    # Đọc file dạng pipe '|' không lấy header
    df = pd.read_csv(
        dat_path, 
        sep='|', 
        comment='#', 
        header=None, 
        skipinitialspace=True, 
        dtype=str, 
        low_memory=False
    )

    print(f" -> Tổng số dòng dữ liệu đọc được: {len(df)}")
    print(f" -> Số cột phát hiện trên mỗi dòng: {df.shape[1]}")

    # Chỉ số cột theo chuẩn CDS/HEASARC Tycho-2:
    # Col 0: TYC ID ("0001 00008 1")
    # Col 2: RAmdeg (2.31750494)
    # Col 3: DEmdeg (2.23184345)
    # Col 17: BTmag (12.146)
    # Col 19: VTmag (12.146)
    
    col_id = 0
    col_ra = 2
    col_dec = 3
    col_bt = 17
    col_vt = 19

    # Trích xuất các cột chính
    extracted = pd.DataFrame({
        'star_id': df[col_id].str.strip(),
        'ra_deg': pd.to_numeric(df[col_ra].str.strip(), errors='coerce'),
        'dec_deg': pd.to_numeric(df[col_dec].str.strip(), errors='coerce'),
        'bt_mag': pd.to_numeric(df[col_bt].str.strip(), errors='coerce'),
        'vt_mag': pd.to_numeric(df[col_vt].str.strip(), errors='coerce')
    })

    # Loại bỏ dòng thiếu tọa độ hoặc magnitude
    extracted = extracted.dropna(subset=['ra_deg', 'dec_deg', 'vt_mag']).copy()

    # Tính toán Cấp sao Quang học Johnson V
    has_bt = extracted['bt_mag'].notna()
    extracted['magnitude'] = np.where(
        has_bt,
        extracted['vt_mag'] - 0.090 * (extracted['bt_mag'] - extracted['vt_mag']),
        extracted['vt_mag']
    )

    # Lọc sao m_V <= max_mag
    df_clean = extracted[extracted['magnitude'] <= max_mag].copy().reset_index(drop=True)

    # Tính toán Vectơ đơn vị 3D ECI
    ra_rad = np.radians(df_clean['ra_deg'].values)
    dec_rad = np.radians(df_clean['dec_deg'].values)

    df_clean['v_eci_x'] = np.cos(dec_rad) * np.cos(ra_rad)
    df_clean['v_eci_y'] = np.cos(dec_rad) * np.sin(ra_rad)
    df_clean['v_eci_z'] = np.sin(dec_rad)

    # Tính Cường độ điểm ảnh cơ sở (80 - 255 DN)
    min_m = df_clean['magnitude'].min() if len(df_clean) > 0 else 1.0
    df_clean['base_intensity'] = np.clip(
        255.0 * (10.0 ** (-0.4 * (df_clean['magnitude'] - min_m))), 
        80.0, 
        255.0
    )

    # Định dạng các cột đầu ra chuẩn
    out_cols = ['star_id', 'ra_deg', 'dec_deg', 'magnitude', 'v_eci_x', 'v_eci_y', 'v_eci_z', 'base_intensity']
    result = df_clean[out_cols]

    result.to_csv(output_csv, index=False)

    print("\n=== XỬ LÝ TYCHO-2 THÀNH CÔNG ===")
    print(f" * Số sao trích xuất (m_V <= {max_mag}): {len(result)}")
    print(f" * File CSV đầu ra                    : {output_csv}\n")

    return output_csv


if __name__ == "__main__":
    # Thay 'tycho2.dat' bằng tên file thực tế của bạn
    convert_tycho2_pipe_to_csv(
        dat_path="catalog.dat",
        output_csv="tycho2_star_catalog.csv",
        max_mag=6.5  # Thay đổi thành 13.0 nếu muốn giữ lại cả những sao mờ như TYC 1-8-1
    )