# =============================================================================
# STAR TRACKER CENTROIDING PIPELINE FOR OPENMV (MICROPYTHON PORT)
# =============================================================================
# Ported directly from the 4-month research codebase image_processing.py
# Optimized for memory limits (no NumPy/SciPy) on STM32 ARM Cortex-M7.
# =============================================================================

import image
import math

# Default configuration parameters (Matched to image_processing.py)
DEFAULT_CONFIG = {
    "MIN_OBJ_AREA": 5,          # Minimum pixel area for a valid object
    "MAX_STAR_AREA": 1000,      # Objects larger than this are candidate outliers
    "K_SENSITIVITY": 2.5,       # Background-relative thresholding factor
    "MIN_LOCAL_CONTRAST": 5,    # Min peak-to-background difference to accept centroid
    "MAD_THRESHOLD": 3.0,       # Robust Z-score limit for planet flagging
    "MIN_SEPARATION_PX": 12.0,  # NMS separation threshold
    "MAX_ASPECT_RATIO": 4.0,     # Max elongation filter
    "LK_MAX_ITER": 10,           # Max refinement iterations (window-shifting)
    "LK_EPSILON": 0.01,          # Convergence delta in pixels
    "BACKGROUND_SCALE": 1/16,    # Scale for global background map estimation
    "EDGE_MARGIN": 5,            # Border margin
    "MIN_PEAK": 10.0             # Minimum peak intensity
}

def get_median(lst):
    """Pure Python median calculation for MicroPython (no numpy.median)."""
    if not lst:
        return 0.0
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 1:
        return sorted_lst[n // 2]
    else:
        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2.0

def get_mad_outliers(data, threshold=3.0):
    """Robust Z-score outliers using Median Absolute Deviation (MAD)."""
    if not data:
        return [False] * len(data)
    median = get_median(data)
    diff = [abs(x - median) for x in data]
    mad = get_median(diff)
    if mad == 0:
        return [False] * len(data)
    z_scores = [0.6745 * d / mad for d in diff]
    return [z > threshold for z in z_scores]

def identify_outliers(objs, config=DEFAULT_CONFIG):
    """
    STARKILLER STAGE 3: Robust Outlier/Planet Identification.
    Matches image_processing.py identify_outliers function.
    """
    if not objs:
        return [], []
        
    compactness = [o['compactness'] for o in objs]
    fluxes = [o['flux'] for o in objs]
    peaks = [o['peak'] for o in objs]
    
    mad_thresh = config.get("MAD_THRESHOLD", 3.0)
    is_size_outlier = get_mad_outliers(compactness, threshold=mad_thresh)
    is_flux_outlier = get_mad_outliers(fluxes, threshold=mad_thresh + 2.0)
    
    planets = []
    stars = []
    
    for i, obj in enumerate(objs):
        is_planet = (is_size_outlier[i] and (peaks[i] > 250)) or (is_size_outlier[i] and is_flux_outlier[i])
        if is_planet:
            planets.append(obj)
        else:
            stars.append(obj)
            
    # Sort by brightness (flux)
    stars.sort(key=lambda x: x['flux'], reverse=True)
    planets.sort(key=lambda x: x['flux'], reverse=True)
    
    return planets, stars

def process_image(img, config=None):
    """
    Processes an OpenMV image, running the mathematically identical 
    Stage 1, 2, 3 and 4 pipeline from image_processing.py.
    """
    cfg = config if config else DEFAULT_CONFIG
    image_width = img.width()
    image_height = img.height()
    
    # Check if color or grayscale
    is_color = (img.format() == image.RGB565)
    
    # Preserve grayscale copy for pipeline processing
    if is_color:
        gray_img = img.copy().to_grayscale() # Convert to grayscale
    else:
        gray_img = img.copy()
        
    # -------------------------------------------------------------------------
    # STAGE 1 — GLOBAL GRAYSCALE PIPELINE (BG estimation & subtraction)
    # -------------------------------------------------------------------------
    blur = gray_img.copy()
    blur.gaussian(1) # Apply Gaussian Blur (sigma=1.0)
    
    # Downscale for background map estimation
    scale = cfg.get("BACKGROUND_SCALE", 1/16)
    small_bg = blur.scale(x_scale=scale, y_scale=scale, hint=image.NEAREST)
    small_bg.median(1) # 3x3 Median Filter
    
    # Upscale back to original size via Bilinear interpolation
    background = small_bg.scale(x_size=image_width, y_size=image_height, hint=image.BILINEAR)
    
    # Subtract background
    clean = blur.copy()
    clean.sub(background)
    
    # Adaptive thresholding statistics
    stats = clean.statistics()
    mean = stats.mean()
    std = stats.stdev()
    k = cfg.get("K_SENSITIVITY", 2.5)
    
    threshold_val = int(mean + k * std)
    if threshold_val > 255:
        threshold_val = 255
        
    # Binarization and Morphology Opening (equivalent to cv2.threshold + Morph Open)
    binary = clean.copy()
    binary.binary([(threshold_val, 255)])
    binary.erode(1)
    binary.dilate(1) # Erode + Dilate is equivalent to Morphological Opening
    
    # Find contours (blobs) on binary image
    contours = binary.find_blobs([(255, 255)], pixels_threshold=cfg.get("MIN_OBJ_AREA", 5), merge=True)
    
    centroids = []
    MIN_LOCAL_CONTRAST = cfg.get("MIN_LOCAL_CONTRAST", 5)
    MAX_ASPECT_RATIO = cfg.get("MAX_ASPECT_RATIO", 4.0)
    LK_MAX_ITER = cfg.get("LK_MAX_ITER", 10)
    LK_EPSILON = cfg.get("LK_EPSILON", 0.01)
    EDGE_MARGIN = cfg.get("EDGE_MARGIN", 5)
    
    for cnt in contours:
        x, y, w, h = cnt.x(), cnt.y(), cnt.w(), cnt.h()
        
        # Dynamic padding proportional to size
        pad = max(3, int(max(w, h) * 0.5))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_width, x + w + pad), min(image_height, y + h + pad)
        
        if w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)
            if aspect > MAX_ASPECT_RATIO:
                continue
                
        # 1. Compute Local Contrast (i_core - i_ring)
        roi_intensity_max = 0
        border_sum = 0
        border_count = 0
        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w <= 0 or roi_h <= 0:
            continue
            
        # Extract max and border pixels directly from the blur image to save memory
        for ry in range(y1, y2):
            for rx in range(x1, x2):
                val = blur.get_pixel(rx, ry)
                if val > roi_intensity_max:
                    roi_intensity_max = val
                # Border check
                if rx == x1 or rx == x2 - 1 or ry == y1 or ry == y2 - 1:
                    border_sum += val
                    border_count += 1
                    
        i_core = roi_intensity_max
        i_ring = border_sum / border_count if border_count > 0 else 0
        
        if (i_core - i_ring) < MIN_LOCAL_CONTRAST:
            continue
            
        # 2. TCG Centroiding (Intensity-weighted center of mass)
        sum_tcg = 0.0
        sum_tcg_x = 0.0
        sum_tcg_y = 0.0
        for ry in range(y1, y2):
            for rx in range(x1, x2):
                val = blur.get_pixel(rx, ry)
                val_tcg = max(val - i_ring, 0.0)
                sum_tcg += val_tcg
                sum_tcg_x += val_tcg * rx
                sum_tcg_y += val_tcg * ry
                
        if sum_tcg == 0:
            continue
            
        cx = sum_tcg_x / sum_tcg
        cy = sum_tcg_y / sum_tcg
        
        # 3. Iterative Refinement (Lucas-Kanade style window shifting)
        HALF_WIN = max(3, int(max(w, h) * 0.5 + 2))
        win_max = 0
        win_tcg_area = 0
        
        for _ in range(LK_MAX_ITER):
            wx1 = int(max(0, cx - HALF_WIN))
            wx2 = int(min(image_width, cx + HALF_WIN + 1))
            wy1 = int(max(0, cy - HALF_WIN))
            wy2 = int(min(image_height, cy + HALF_WIN + 1))
            
            win_w = wx2 - wx1
            win_h = wy2 - wy1
            if win_w <= 2 or win_h <= 2:
                break
                
            # Compute border background of shifted window
            b_sum = 0
            b_count = 0
            win_max = 0
            for wy in range(wy1, wy2):
                for wx in range(wx1, wx2):
                    val = blur.get_pixel(wx, wy)
                    if val > win_max:
                        win_max = val
                    if wx == wx1 or wx == wx2 - 1 or wy == wy1 or wy == wy2 - 1:
                        b_sum += val
                        b_count += 1
            win_bg = b_sum / b_count if b_count > 0 else 0
            
            # Recalculate center of mass
            s = 0.0
            wx_sum = 0.0
            wy_sum = 0.0
            win_tcg_area = 0
            
            for wy in range(wy1, wy2):
                for wx in range(wx1, wx2):
                    val = blur.get_pixel(wx, wy)
                    val_tcg = max(val - win_bg, 0.0)
                    if val_tcg > 0:
                        win_tcg_area += 1
                    s += val_tcg
                    wx_sum += val_tcg * wx
                    wy_sum += val_tcg * wy
                    
            if s == 0:
                break
                
            new_cx = wx_sum / s
            new_cy = wy_sum / s
            shift = math.sqrt((new_cx - cx)**2 + (new_cy - cy)**2)
            cx, cy = new_cx, new_cy
            if shift < LK_EPSILON:
                break
                
        # 4. Check boundaries and minimum peak
        if (cx < EDGE_MARGIN or cy < EDGE_MARGIN or 
            cx > image_width - EDGE_MARGIN or cy > image_height - EDGE_MARGIN or
            win_max < cfg.get("MIN_PEAK", 10.0)):
            continue
            
        centroids.append({
            "x": cx, 
            "y": cy, 
            "flux": float(sum_tcg), 
            "peak": float(win_max), 
            "area": int(win_tcg_area), 
            "type": "gray"
        })

    # -------------------------------------------------------------------------
    # STAGE 2 — MULTI-CHANNEL COLOR-SHIFTED DETECTION
    # -------------------------------------------------------------------------
    new_color_centroids = []
    MIN_SEPARATION_PX = cfg.get("MIN_SEPARATION_PX", 12.0)
    
    if is_color:
        # Loop through red (0), green (1), and blue (2) channels
        for ch_idx in range(3):
            # Extract color channel as grayscale buffer
            ch = img.copy().to_grayscale(rgb_channel=ch_idx)
            ch_blur = ch.copy()
            ch_blur.gaussian(1)
            
            ch_small = ch_blur.scale(x_scale=scale, y_scale=scale, hint=image.NEAREST)
            ch_small.median(1)
            ch_bg = ch_small.scale(x_size=image_width, y_size=image_height, hint=image.BILINEAR)
            
            ch_clean = ch_blur.copy()
            ch_clean.sub(ch_bg)
            
            ch_stats = ch_clean.statistics()
            ch_mean = ch_stats.mean()
            ch_std = ch_stats.stdev()
            
            ch_thresh = int(ch_mean + 3.5 * ch_std)
            if ch_thresh > 255:
                ch_thresh = 255
                
            ch_bin = ch_clean.copy()
            ch_bin.binary([(ch_thresh, 255)])
            ch_bin.erode(1)
            ch_bin.dilate(1)
            
            ch_contours = ch_bin.find_blobs([(255, 255)], pixels_threshold=cfg.get("MIN_OBJ_AREA", 5), merge=True)
            
            for cnt in ch_contours:
                cx2, cy2, w2, h2 = cnt.x(), cnt.y(), cnt.w(), cnt.h()
                if w2 == 0 or h2 == 0 or max(w2, h2) / min(w2, h2) > cfg.get("MAX_ASPECT_RATIO", 4.0):
                    continue
                pad2 = max(2, int(max(w2, h2) * 0.4))
                bx1, by1 = max(0, cx2 - pad2), max(0, cy2 - pad2)
                bx2, by2 = min(image_width, cx2 + w2 + pad2), min(image_height, cy2 + h2 + pad2)
                
                roi_w2 = bx2 - bx1
                roi_h2 = by2 - by1
                if roi_w2 <= 0 or roi_h2 <= 0:
                    continue
                    
                i_core_ch = 0
                b_sum2 = 0
                b_count2 = 0
                for ry in range(by1, by2):
                    for rx in range(bx1, bx2):
                        val = ch_clean.get_pixel(rx, ry)
                        if val > i_core_ch:
                            i_core_ch = val
                        if rx == bx1 or rx == bx2 - 1 or ry == by1 or ry == by2 - 1:
                            b_sum2 += val
                            b_count2 += 1
                i_ring_ch = b_sum2 / b_count2 if b_count2 > 0 else 0
                
                if (i_core_ch - i_ring_ch) < MIN_LOCAL_CONTRAST:
                    continue
                    
                # Compute TCG on color channel
                s2 = 0.0
                sum_cx = 0.0
                sum_cy = 0.0
                ch_area = 0
                for ry in range(by1, by2):
                    for rx in range(bx1, bx2):
                        val = ch_clean.get_pixel(rx, ry)
                        val_tcg = max(val - i_ring_ch, 0.0)
                        if val_tcg > 0:
                            ch_area += 1
                        s2 += val_tcg
                        sum_cx += val_tcg * rx
                        sum_cy += val_tcg * ry
                        
                if s2 == 0:
                    continue
                ncx = sum_cx / s2
                ncy = sum_cy / s2
                
                # Check separation against existing grayscale centroids (KDTree query replacement)
                is_new = True
                for g_pt in centroids:
                    d = math.sqrt((g_pt['x'] - ncx)**2 + (g_pt['y'] - ncy)**2)
                    if d < MIN_SEPARATION_PX:
                        is_new = False
                        break
                if is_new:
                    # Check separation against already added color centroids
                    for c_pt in new_color_centroids:
                        d = math.sqrt((c_pt['x'] - ncx)**2 + (c_pt['y'] - ncy)**2)
                        if d < MIN_SEPARATION_PX:
                            is_new = False
                            break
                            
                if not is_new:
                    continue
                    
                # Border check and peak check
                if (ncx < EDGE_MARGIN or ncy < EDGE_MARGIN or 
                    ncx > image_width - EDGE_MARGIN or ncy > image_height - EDGE_MARGIN or
                    i_core_ch < cfg.get("MIN_PEAK", 10.0)):
                    continue
                    
                new_color_centroids.append({
                    "x": ncx, 
                    "y": ncy, 
                    "flux": float(s2), 
                    "peak": float(i_core_ch), 
                    "area": int(ch_area), 
                    "type": "color_" + str(ch_idx)
                })

    # -------------------------------------------------------------------------
    # STAGE 3 — CONSOLIDATION & NON-MAXIMA SUPPRESSION
    # -------------------------------------------------------------------------
    all_objs = centroids + new_color_centroids
    all_objs.sort(key=lambda x: x['peak'], reverse=True)
    
    final_objs = []
    for obj in all_objs:
        keep = True
        for f in final_objs:
            d = math.sqrt((obj['x'] - f['x'])**2 + (obj['y'] - f['y'])**2)
            if d < MIN_SEPARATION_PX:
                keep = False
                break
        if keep:
            final_objs.append(obj)
            
    # -------------------------------------------------------------------------
    # STAGE 4 — OUTLIER DETECTION (MAD Filtering)
    # -------------------------------------------------------------------------
    for o in final_objs:
        o['compactness'] = o['flux'] / (o['peak'] + 1e-6)
        
    planets, stars = identify_outliers(final_objs, config=cfg)
    
    # Return detected stars and planetary outliers separately
    return stars, planets
