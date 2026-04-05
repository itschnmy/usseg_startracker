import cv2
import numpy as np
import os
import time
import json
from pathlib import Path
from scipy.spatial import KDTree

# =============================================================================
# STAR TRACKER IMAGE PROCESSING PIPELINE
# =============================================================================

# --- CONFIGURABLE PARAMETERS ---
# Default configuration parameters for the star detection and outlier filtering pipeline.
# These can be overridden by passing a config dictionary to process_single_image().
DEFAULT_CONFIG = {
    "MIN_OBJ_AREA": 5,      # Minimum pixel area for a valid object (reduces shot noise)
    "MAX_STAR_AREA": 1000,  # Objects larger than this are candidates for planetary outliers
    "K_SENSITIVITY": 2.5,   # Sensitivity multiplier (k) for background-relative thresholding
    "MIN_LOCAL_CONTRAST": 5,# Min local peak-to-background difference to accept a centroid
    "MAD_THRESHOLD": 3.0,   # Robust Z-score limit (Median Absolute Deviation) for planet flagging
    "MIN_SEPARATION_PX": 12.0, # Consolidate detections within this radius
    "MAX_ASPECT_RATIO": 4.0,   # Filters out extremely elongated objects (traces, sensor defects)
    "LK_MAX_ITER": 10,         # Maximum refinement iterations for sub-pixel centroiding
    "LK_EPSILON": 0.01,        # Sub-pixel convergence threshold in pixels
    "BACKGROUND_SCALE": 1/32,  # Resolution scale (fx, fy) for estimating the global background map
    "EDGE_MARGIN": 5,           # Ignore detections near the sensor boundary
    "MIN_PEAK": 10.0            # Minimum peak intensity to consider a star real
}

def identify_outliers(objs, config=None):
    """
    STARKILLER STAGE 3: Statistical Outlier Identification.
    Checks the Point Spread Function (PSF) compactness (Flux / Peak) against the median 
    profile of the field using Robust Z-scores (MAD). 
    Objects with significantly higher compactness are likely resolved planetary disks, 
    saturated blooms, or hot pixels.
    """
    cfg = config if config else DEFAULT_CONFIG
    if not objs: return [], []
    
    compactness = np.array([o['compactness'] for o in objs])
    fluxes      = np.array([o['flux'] for o in objs])
    peaks       = np.array([o['peak'] for o in objs])

    def get_mad_outliers(data, threshold=3.0):
        median = np.median(data)
        diff = np.abs(data - median)
        mad = np.median(diff)
        if mad == 0: return np.zeros(len(data), dtype=bool)
        z_scores = 0.6745 * diff / mad
        return np.abs(z_scores) > threshold

    mad_thresh = cfg.get("MAD_THRESHOLD", 3.0)
    is_size_outlier = get_mad_outliers(compactness, threshold=mad_thresh)
    is_flux_outlier = get_mad_outliers(fluxes, threshold=mad_thresh + 2.0) 
    
    is_planet = (is_size_outlier & (peaks > 250)) | (is_size_outlier & is_flux_outlier)
    
    planets = [objs[i] for i in range(len(objs)) if is_planet[i]]
    stars   = [objs[i] for i in range(len(objs)) if not is_planet[i]]
    
    # Sort by brightness
    stars.sort(key=lambda x: x['flux'], reverse=True)
    planets.sort(key=lambda x: x['flux'], reverse=True)
    
    return planets, stars

def process_single_image(img_path, img_adjust_dir, config=None):
    """
    Processes a single image file, detects stars/outliers, saves visualizations and JSON.
    Returns: (stars, planets, execution_time_ms)
    """
    cfg = config if config else DEFAULT_CONFIG
    start_time = time.time()
    img_path = Path(img_path)
    img_adjust_dir = Path(img_adjust_dir)
    img_adjust_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        return None, None, 0

    # Preserve original color image for verification
    raw_image_color = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # STAGE 1 — GLOBAL GRAYSCALE PIPELINE
    # Uses a high-gain background subtraction to find faint stellar candidates.
    # -------------------------------------------------------------------------
    blur = cv2.GaussianBlur(image, (3, 3), 1.0)
    scale = cfg.get("BACKGROUND_SCALE", 1/16)
    small_bg = cv2.resize(blur, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    small_bg = cv2.medianBlur(small_bg, 3)
    background = cv2.resize(small_bg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    clean = cv2.subtract(blur, background)

    mean = np.mean(clean)
    std  = np.std(clean)
    k    = cfg.get("K_SENSITIVITY", 2.5)

    _, binary = cv2.threshold(clean, mean + k * std, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_LOCAL_CONTRAST = cfg.get("MIN_LOCAL_CONTRAST", 5)
    MAX_ASPECT_RATIO = cfg.get("MAX_ASPECT_RATIO", 4.0)

    # Iterative centroiding convergence parameters (LK-style window shifting)
    LK_MAX_ITER = cfg.get("LK_MAX_ITER", 10)
    LK_EPSILON  = cfg.get("LK_EPSILON", 0.01)  # Convergence threshold in pixels

    EDGE_MARGIN = cfg.get("EDGE_MARGIN", 5)

    centroids = []
    clean_vis = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug_vis = cv2.cvtColor(clean_vis, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Dynamic padding proportional to contour size
        pad = max(3, int(max(w, h) * 0.5))

        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)

        if w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)
            if aspect > MAX_ASPECT_RATIO: continue

        roi_intensity = blur[y1:y2, x1:x2].astype(float)
        if roi_intensity.size == 0: continue

        i_core = np.max(roi_intensity)
        border_mask = np.ones(roi_intensity.shape, dtype=bool)
        if roi_intensity.shape[0] > 2 and roi_intensity.shape[1] > 2:
            border_mask[1:-1, 1:-1] = False
        i_ring = np.mean(roi_intensity[border_mask])

        if (i_core - i_ring) < MIN_LOCAL_CONTRAST: continue

        Y_len, X_len = roi_intensity.shape
        x_vec = np.arange(X_len, dtype=np.float32)
        y_vec = np.arange(Y_len, dtype=np.float32)
        roi_tcg = np.maximum(roi_intensity - i_ring, 0.0)
        sum_tcg = np.sum(roi_tcg)
        if sum_tcg == 0: continue

        cx = x1 + np.dot(np.sum(roi_tcg, axis=0), x_vec) / sum_tcg
        cy = y1 + np.dot(np.sum(roi_tcg, axis=1), y_vec) / sum_tcg

        HALF_WIN = max(3, int(max(w, h) * 0.5 + 2))
        win_tcg = roi_tcg # default if no iter
        win = blur[y1:y2, x1:x2]

        for _ in range(LK_MAX_ITER):
            wx1 = int(max(0, cx - HALF_WIN)); wx2 = int(min(image.shape[1], cx + HALF_WIN + 1))
            wy1 = int(max(0, cy - HALF_WIN)); wy2 = int(min(image.shape[0], cy + HALF_WIN + 1))
            win = blur[wy1:wy2, wx1:wx2].astype(np.float32)
            if win.size == 0: break
            bm = np.ones(win.shape, dtype=bool)
            if win.shape[0] > 2 and win.shape[1] > 2: bm[1:-1, 1:-1] = False
            win_bg = np.mean(win[bm])
            win_tcg = np.maximum(win - win_bg, 0.0)
            s = np.sum(win_tcg)
            if s == 0: break
            wx_vec = np.arange(win_tcg.shape[1], dtype=np.float32)
            wy_vec = np.arange(win_tcg.shape[0], dtype=np.float32)
            new_cx = wx1 + np.dot(np.sum(win_tcg, axis=0), wx_vec) / s
            new_cy = wy1 + np.dot(np.sum(win_tcg, axis=1), wy_vec) / s
            shift = np.hypot(new_cx - cx, new_cy - cy)
            cx, cy = new_cx, new_cy
            if shift < LK_EPSILON: break

        # Skip detections too close to the edge or below peak threshold
        if (cx < EDGE_MARGIN or cy < EDGE_MARGIN or 
            cx > image.shape[1] - EDGE_MARGIN or cy > image.shape[0] - EDGE_MARGIN or
            np.max(win) < cfg.get("MIN_PEAK", 10.0)):
            continue

        obj_area = np.count_nonzero(win_tcg)
        centroids.append({"x": cx, "y": cy, "flux": float(sum_tcg), "peak": float(np.max(win)), "area": int(obj_area), "type": "gray"})

        cx_i, cy_i = int(round(cx)), int(round(cy))
        vh = max(4, int(max(w, h) * 0.5 + pad))
        cv2.rectangle(debug_vis, (max(0, cx_i - vh), max(0, cy_i - vh)), (min(image.shape[1], cx_i + vh), min(image.shape[0], cy_i + vh)), (0, 255, 0), 1)
        cv2.circle(debug_vis, (cx_i, cy_i), 2, (0, 0, 255), -1)
        cv2.rectangle(raw_image_color, (max(0, cx_i - vh), max(0, cy_i - vh)), (min(image.shape[1], cx_i + vh), min(image.shape[0], cy_i + vh)), (0, 255, 0), 1)
        cv2.drawMarker(raw_image_color, (cx_i, cy_i), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    # -------------------------------------------------------------------------
    # STAGE 2 — MULTI-CHANNEL COLOR-SHIFTED DETECTION
    # -------------------------------------------------------------------------
    MIN_SEPARATION_PX = cfg.get("MIN_SEPARATION_PX", 10.0)
    CHANNEL_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    new_color_centroids = []
    gray_tree = KDTree([(o['x'], o['y']) for o in centroids]) if centroids else None
    color_pts_for_tree = []

    for ch_idx in range(3):
        ch = raw_image_color[:, :, ch_idx].astype(np.float32)
        ch_blur = cv2.GaussianBlur(ch, (3, 3), 1.0)
        scale = cfg.get("BACKGROUND_SCALE", 1/16)
        ch_small = cv2.resize(ch_blur, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ch_bg = cv2.resize(cv2.medianBlur(ch_small.astype(np.uint8), 3).astype(np.float32), (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_LINEAR)
        ch_clean = np.maximum(ch_blur - ch_bg, 0).astype(np.float32)
        ch_mean, ch_std = np.mean(ch_clean), np.std(ch_clean)
        _, ch_bin = cv2.threshold(ch_clean.astype(np.uint8), int(ch_mean + 3.5 * ch_std), 255, cv2.THRESH_BINARY)
        ch_bin = cv2.morphologyEx(ch_bin, cv2.MORPH_OPEN, kernel)
        ch_contours, _ = cv2.findContours(ch_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in ch_contours:
            cx2, cy2, w2, h2 = cv2.boundingRect(cnt)
            if w2 == 0 or h2 == 0 or max(w2, h2) / min(w2, h2) > cfg.get("MAX_ASPECT_RATIO", 4.0): continue
            pad2 = max(2, int(max(w2, h2) * 0.4))
            bx1 = max(0, cx2 - pad2); bx2 = min(ch.shape[1], cx2 + w2 + pad2)
            by1 = max(0, cy2 - pad2); by2 = min(ch.shape[0], cy2 + h2 + pad2)
            roi_ch = ch_clean[by1:by2, bx1:bx2]
            if roi_ch.size == 0: continue
            i_core_ch = np.max(roi_ch)
            bm2 = np.ones(roi_ch.shape, dtype=bool)
            if roi_ch.shape[0] > 2 and roi_ch.shape[1] > 2: bm2[1:-1, 1:-1] = False
            i_ring_ch = np.mean(roi_ch[bm2])
            if (i_core_ch - i_ring_ch) < MIN_LOCAL_CONTRAST: continue
            roi_tcg2 = np.maximum(roi_ch - i_ring_ch, 0.0)
            s2 = np.sum(roi_tcg2)
            if s2 == 0: continue
            ncx = bx1 + np.dot(np.sum(roi_tcg2, axis=0), np.arange(roi_tcg2.shape[1])) / s2
            ncy = by1 + np.dot(np.sum(roi_tcg2, axis=1), np.arange(roi_tcg2.shape[0])) / s2
            
            is_new = True
            if gray_tree is not None:
                dist, _ = gray_tree.query([[ncx, ncy]], k=1)
                if dist[0] < MIN_SEPARATION_PX: is_new = False
            if is_new and color_pts_for_tree:
                dists = np.hypot(np.array(color_pts_for_tree)[:,0] - ncx, np.array(color_pts_for_tree)[:,1] - ncy)
                if np.min(dists) < MIN_SEPARATION_PX: is_new = False
            
            if not is_new: continue
            
            # Skip edge detections or low peak objects in color channels
            if (ncx < EDGE_MARGIN or ncy < EDGE_MARGIN or 
                ncx > ch.shape[1] - EDGE_MARGIN or ncy > ch.shape[0] - EDGE_MARGIN or
                i_core_ch < cfg.get("MIN_PEAK", 10.0)):
                continue
                
            new_color_centroids.append({"x": ncx, "y": ncy, "flux": float(s2), "peak": float(i_core_ch), "area": int(np.count_nonzero(roi_tcg2)), "type": f"color_{ch_idx}"})
            color_pts_for_tree.append([ncx, ncy])
            col = CHANNEL_COLORS[ch_idx]; ncx_i, ncy_i = int(round(ncx)), int(round(ncy)); vh = max(4, int(max(w2, h2) * 0.5 + pad2))
            cv2.rectangle(debug_vis, (max(0, ncx_i - vh), max(0, ncy_i - vh)), (min(image.shape[1], ncx_i + vh), min(image.shape[0], ncy_i + vh)), col, 1)
            cv2.rectangle(raw_image_color, (max(0, ncx_i - vh), max(0, ncy_i - vh)), (min(raw_image_color.shape[1], ncx_i + vh), min(raw_image_color.shape[0], ncy_i + vh)), col, 1)
            cv2.drawMarker(raw_image_color, (ncx_i, ncy_i), color=col, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)

    # -------------------------------------------------------------------------
    # STAGE 3 — CONSOLIDATION & NMS
    # Sorts by peak intensity and removes weaker overlapping detections.
    # -------------------------------------------------------------------------
    all_objs = centroids + new_color_centroids
    all_objs.sort(key=lambda x: x['peak'], reverse=True)
    
    final_objs = []
    MIN_SEP = cfg.get("MIN_SEPARATION_PX", 12.0)
    for obj in all_objs:
        keep = True
        for f in final_objs:
            if np.hypot(obj['x'] - f['x'], obj['y'] - f['y']) < MIN_SEP:
                keep = False
                break
        if keep: final_objs.append(obj)
    
    # -------------------------------------------------------------------------
    # STAGE 4 — OUTLIER DETECTION
    # -------------------------------------------------------------------------
    for o in final_objs: o['compactness'] = o['flux'] / (o['peak'] + 1e-6)
    planets, stars = identify_outliers(final_objs, config=cfg)
    sorted_objs = stars + planets
    all_centroids_xy = [(o['x'], o['y']) for o in sorted_objs]

    # Export JSON
    export_data = {
        "image_name": img_path.name,
        "image_size_hw": [int(image.shape[0]), int(image.shape[1])],
        "centroids_xy": [[float(o['x']), float(o['y'])] for o in sorted_objs],
        "objects": [{
            "x": float(o['x']), 
            "y": float(o['y']), 
            "flux": float(o['flux']), 
            "peak": float(o['peak']), 
            "area": int(o['area']), 
            "type": o['type'],
            "is_outlier": o in planets
        } for o in sorted_objs],
        "outliers_count": int(len(planets)), 
        "stars_count": int(len(stars)), 
        "total_count": int(len(sorted_objs)),
        "outlier_indices": [int(i) for i, o in enumerate(sorted_objs) if o in planets]
    }
    with open(img_adjust_dir / "centroids.json", "w") as jf: json.dump(export_data, jf, indent=2)

    # Visualizations
    vis5 = cv2.imread(str(img_path))
    if vis5 is None: vis5 = raw_image_color.copy()
    BOX_COLOR = (0, 255, 0); OUTLIER_COL = (0, 0, 255); MARK_COLOR = (255, 255, 255); BOX_HALF = 8
    for i, (cx_f, cy_f) in enumerate(all_centroids_xy):
        is_obj_planet = (i >= len(stars)); ci, cj = int(round(cx_f)), int(round(cy_f))
        vx1b, vy1b = max(0, ci - BOX_HALF), max(0, cj - BOX_HALF)
        vx2b, vy2b = min(vis5.shape[1] - 1, ci + BOX_HALF), min(vis5.shape[0] - 1, cj + BOX_HALF)
        color = OUTLIER_COL if is_obj_planet else BOX_COLOR
        cv2.rectangle(vis5, (vx1b, vy1b), (vx2b, vy2b), color, 1)
        if is_obj_planet:
            cv2.line(vis5, (vx1b, vy1b), (vx2b, vy2b), color, 1); cv2.line(vis5, (vx1b, vy2b), (vx2b, vy1b), color, 1)
        else:
            cv2.drawMarker(vis5, (ci, cj), color=MARK_COLOR, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    vis6 = cv2.imread(str(img_path))
    if vis6 is None: vis6 = raw_image_color.copy()
    for o in stars:
        ci, cj = int(round(o['x'])), int(round(o['y']))
        vx1b, vy1b = max(0, ci - BOX_HALF), max(0, cj - BOX_HALF)
        vx2b, vy2b = min(vis6.shape[1] - 1, ci + BOX_HALF), min(vis6.shape[0] - 1, cj + BOX_HALF)
        cv2.rectangle(vis6, (vx1b, vy1b), (vx2b, vy2b), BOX_COLOR, 1)
        cv2.drawMarker(vis6, (ci, cj), color=MARK_COLOR, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    cv2.imwrite(str(img_adjust_dir / "1_clean.png"), clean)
    cv2.imwrite(str(img_adjust_dir / "2_binary.png"), binary)
    cv2.imwrite(str(img_adjust_dir / "3_debug_centroids.png"), debug_vis)
    cv2.imwrite(str(img_adjust_dir / "4_human_verify.png"), raw_image_color)
    cv2.imwrite(str(img_adjust_dir / "5_visualization.png"), vis5)
    cv2.imwrite(str(img_adjust_dir / "6_filtered_stars.png"), vis6)

    inference_time = (time.time() - start_time) * 1000
    return stars, planets, inference_time

if __name__ == "__main__":
    image_dir = Path("starimage"); adjust_root = image_dir / "adjust"; adjust_root.mkdir(exist_ok=True)
    images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.bmp', '.png', '.jpg', '.jpeg'}]
    for img in images:
        print(f"Processing: {img.name}"); process_single_image(img, adjust_root / img.stem)
    print("Done.")