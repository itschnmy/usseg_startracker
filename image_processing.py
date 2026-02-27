import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from scipy.spatial import KDTree

# =============================================================================
# STAR TRACKER IMAGE PROCESSING PIPELINE
# =============================================================================
# Implements a two-stage star centroiding pipeline:
#   Stage 1 – GRAY channel: Background subtraction → Thresholding →
#             Morphological filtering → Iterative LK-style centroiding
#   Stage 2 – Per-channel (RGB): Detects color-shifted celestial objects
#             (redshift / blueshift) missed by the grayscale pipeline.
#
# =============================================================================

# -----------------------------------------------------------------------------
# INPUT / OUTPUT PATHS
# -----------------------------------------------------------------------------
image_dir = Path(r"your/absolute/path/to/image/folder")

# Output root directory for all processed results
adjust_root = image_dir / "adjust"
adjust_root.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}
image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

if not image_files:
    raise ValueError("No image files found in the specified directory.")

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
for img_path in image_files:
    print(f"\n{'='*60}")
    print(f"  Processing : {img_path.name}")
    print(f"{'='*60}")
    start_time = time.time()

    # Create a dedicated output sub-directory for this image
    img_adjust_dir = adjust_root / img_path.stem
    img_adjust_dir.mkdir(exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [WARNING] Could not read: {img_path.name} — skipping.")
        continue

    # Preserve original color image for human verification and color-channel pipeline
    raw_image_color = image.copy()

    # Convert to grayscale to reduce memory footprint
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # STAGE 1 — GRAYSCALE PIPELINE
    # -------------------------------------------------------------------------

    # Step 1a: Gaussian smoothing
    blur = cv2.GaussianBlur(image, (3, 3), 1.0)
    # Step 1b: Background estimation via downscaling
    # Downscale factor 1/16 → background image is 256× smaller than original.
    scale = 1/16
    small_bg = cv2.resize(blur, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    small_bg = cv2.medianBlur(small_bg, 3)  # Kernel size 3 to keep CPU load low
    background = cv2.resize(small_bg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 1c: Subtract estimated background
    clean = cv2.subtract(blur, background)

    # Step 1d: Global thresholding (mean + k * std)
    mean = np.mean(clean)
    std  = np.std(clean)
    k    = 2.5  # Sensitivity coefficient — lower values detect fainter stars

    _, binary = cv2.threshold(
        clean,
        mean + k * std,
        255,
        cv2.THRESH_BINARY
    )

    # Step 1e: Morphological opening to remove isolated noise pixels 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 1f: Connected-component contour detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -------------------------------------------------------------------------
    # CENTROIDING PARAMETERS
    # -------------------------------------------------------------------------
    # Minimum intensity difference between star core and local background ring.
    # Using a small value because DoG-style subtraction already attenuates the
    # background, so faint stars still produce a measurable local contrast.
    MIN_LOCAL_CONTRAST = 5

    # Shape filter: reject elongated streaks (satellites, meteors, cosmic rays).
    # Objects with major-axis / minor-axis > MAX_ASPECT_RATIO are discarded.
    MAX_ASPECT_RATIO = 4.0

    # Iterative centroiding convergence parameters (LK-style window shifting)
    LK_MAX_ITER = 10
    LK_EPSILON  = 0.01  # Convergence threshold in pixels

    # Prepare visualisation canvases
    centroids  = []
    clean_vis  = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug_vis  = cv2.cvtColor(clean_vis, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------------------------------
    # ITERATIVE CENTROIDING (LK-style window shifting)
    # -------------------------------------------------------------------------
    # Algorithm overview:
    #   1. Seed estimate: Thresholded Center-of-Gravity (TCG) on the initial
    #      contour bounding-box ROI — eliminates background bias in CoG.
    #   2. Iterative refinement: Re-crop a fixed-size window around the
    #      current centroid estimate, recompute TCG, shift window, repeat
    #      until displacement < LK_EPSILON or max iterations reached.
    #      This converges the estimate toward the true photometric center and
    #      removes the S-curve error inherent to simple CoG centroiding.
    # -------------------------------------------------------------------------
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Dynamic padding proportional to contour size
        pad = max(3, int(max(w, h) * 0.5))

        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)

        # Shape filter: discard elongated streaks
        if w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)
            if aspect > MAX_ASPECT_RATIO:
                continue

        roi_intensity = blur[y1:y2, x1:x2].astype(float)
        if roi_intensity.size == 0:
            continue

        i_core = np.max(roi_intensity)

        # Estimate local background using the border ring of the ROI
        border_mask = np.ones(roi_intensity.shape, dtype=bool)
        border_mask[1:-1, 1:-1] = False
        i_ring = np.mean(roi_intensity[border_mask])

        # Local contrast filter
        if (i_core - i_ring) < MIN_LOCAL_CONTRAST:
            continue

        # Step 1: Seed centroid via Thresholded CoG (TCG)
        Y_len, X_len = roi_intensity.shape
        x_vec   = np.arange(X_len, dtype=np.float32)
        y_vec   = np.arange(Y_len, dtype=np.float32)
        roi_tcg = np.maximum(roi_intensity - i_ring, 0.0)
        sum_tcg = np.sum(roi_tcg)
        if sum_tcg == 0:
            continue

        cx = x1 + np.dot(np.sum(roi_tcg, axis=0), x_vec) / sum_tcg
        cy = y1 + np.dot(np.sum(roi_tcg, axis=1), y_vec) / sum_tcg

        # Step 2: Iterative window-shifting refinement
        HALF_WIN = max(3, int(max(w, h) * 0.5 + 2))  # Fixed window half-size

        for _ in range(LK_MAX_ITER):
            # Crop window centered on current estimate (cx, cy)
            wx1 = int(max(0,              cx - HALF_WIN))
            wx2 = int(min(image.shape[1], cx + HALF_WIN + 1))
            wy1 = int(max(0,              cy - HALF_WIN))
            wy2 = int(min(image.shape[0], cy + HALF_WIN + 1))

            win = blur[wy1:wy2, wx1:wx2].astype(np.float32)
            if win.size == 0:
                break

            # Local background of the current window
            bm = np.ones(win.shape, dtype=bool)
            if win.shape[0] > 2 and win.shape[1] > 2:
                bm[1:-1, 1:-1] = False
            win_bg  = np.mean(win[bm])
            win_tcg = np.maximum(win - win_bg, 0.0)

            s = np.sum(win_tcg)
            if s == 0:
                break

            wx_vec = np.arange(win_tcg.shape[1], dtype=np.float32)
            wy_vec = np.arange(win_tcg.shape[0], dtype=np.float32)

            new_cx = wx1 + np.dot(np.sum(win_tcg, axis=0), wx_vec) / s
            new_cy = wy1 + np.dot(np.sum(win_tcg, axis=1), wy_vec) / s

            # Check convergence
            shift  = np.hypot(new_cx - cx, new_cy - cy)
            cx, cy = new_cx, new_cy
            if shift < LK_EPSILON:
                break

        x_cog, y_cog = cx, cy
        centroids.append((x_cog, y_cog))

        # Draw detection bounding box and centroid marker
        cx_i, cy_i = int(round(x_cog)), int(round(y_cog))
        vis_half = max(4, int(max(w, h) * 0.5 + pad))
        vx1 = max(0, cx_i - vis_half)
        vx2 = min(image.shape[1], cx_i + vis_half)
        vy1 = max(0, cy_i - vis_half)
        vy2 = min(image.shape[0], cy_i + vis_half)

        # Overlay on background-subtracted debug image
        cv2.rectangle(debug_vis, (vx1, vy1), (vx2, vy2), (0, 255, 0), 1)
        cv2.circle(debug_vis, (cx_i, cy_i), 2, (0, 0, 255), -1)

        # Overlay on original color image (for human verification)
        cv2.rectangle(raw_image_color, (vx1, vy1), (vx2, vy2), (0, 255, 0), 1)
        cv2.drawMarker(raw_image_color, (cx_i, cy_i),
                       color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=5, thickness=1)

    # =========================================================================
    # STAGE 2 — MULTI-CHANNEL COLOR-SHIFTED DETECTION
    # =========================================================================
    # Detects celestial objects biased toward a specific color channel that were
    # missed by the grayscale pipeline (e.g., red stars suppressed by the standard
    # GRAY weighting: 0.299R + 0.587G + 0.114B).
    # Inspired by: SExtractor multi-band detection; ZTF transient detection pipeline.
    #
    # Each channel is processed independently with its own background subtraction
    # and thresholding. Candidates that overlap (within MIN_SEPARATION_PX) with
    # already-detected grayscale centroids are discarded to avoid duplicates.
    # Nearest-neighbor lookup uses a KDTree.
    # =========================================================================

    # Minimum pixel separation between a color-channel candidate and any existing centroid
    MIN_SEPARATION_PX = 10.0

    # Visualisation colors per channel (BGR format)
    CHANNEL_COLORS = {
        0: (255, 0,  0),   # Blue  channel → Cyan marker
        1: (0,  255, 0),   # Green channel → Lime marker
        2: (0,  0,  255),  # Red   channel → Deep Red marker
    }
    CHANNEL_NAMES = {0: 'Blue', 1: 'Green', 2: 'Red (Redshifted)'}

    new_color_centroids = []  # List of (ncx, ncy, ch_idx, vis_half_size)

    # Build KDTree from grayscale centroids for fast duplicate rejection
    if len(centroids) > 0:
        gray_pts  = np.array([(cx, cy) for cx, cy in centroids], dtype=np.float32)
        gray_tree = KDTree(gray_pts)
    else:
        gray_tree = None

    color_pts_for_tree = []  # Grows as new color-channel centroids are accepted

    for ch_idx in range(3):
        ch = raw_image_color[:, :, ch_idx].astype(np.float32)

        # Per-channel background subtraction (same approach as Stage 1)
        ch_blur  = cv2.GaussianBlur(ch, (3, 3), 1.0)
        scale    = 1/16
        ch_small = cv2.resize(ch_blur, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ch_small = cv2.medianBlur(ch_small.astype(np.uint8), 3).astype(np.float32)
        ch_bg    = cv2.resize(ch_small, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_LINEAR)
        ch_clean = np.maximum(ch_blur - ch_bg, 0).astype(np.float32)

        # Stricter threshold (k=3.5) to select objects truly dominant in this channel
        ch_mean, ch_std = np.mean(ch_clean), np.std(ch_clean)
        _, ch_bin = cv2.threshold(
            ch_clean.astype(np.uint8),
            int(ch_mean + 3.5 * ch_std),
            255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ch_bin = cv2.morphologyEx(ch_bin, cv2.MORPH_OPEN, kernel)

        ch_contours, _ = cv2.findContours(ch_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in ch_contours:
            cx2, cy2, w2, h2 = cv2.boundingRect(cnt)
            if w2 == 0 or h2 == 0:
                continue
            if max(w2, h2) / min(w2, h2) > 4.0:
                continue  # Reject elongated streaks

            pad2 = max(2, int(max(w2, h2) * 0.4))
            bx1 = max(0, cx2 - pad2);          bx2 = min(ch.shape[1], cx2 + w2 + pad2)
            by1 = max(0, cy2 - pad2);          by2 = min(ch.shape[0], cy2 + h2 + pad2)

            roi_ch = ch_clean[by1:by2, bx1:bx2]
            if roi_ch.size == 0:
                continue

            i_core_ch = np.max(roi_ch)
            bm2 = np.ones(roi_ch.shape, dtype=bool)
            if roi_ch.shape[0] > 2 and roi_ch.shape[1] > 2:
                bm2[1:-1, 1:-1] = False
            i_ring_ch = np.mean(roi_ch[bm2])
            if (i_core_ch - i_ring_ch) < MIN_LOCAL_CONTRAST:
                continue

            # Fast TCG centroiding (no iterative refinement needed at this stage)
            roi_tcg2 = np.maximum(roi_ch - i_ring_ch, 0.0)
            s2 = np.sum(roi_tcg2)
            if s2 == 0:
                continue
            y2_vec = np.arange(roi_tcg2.shape[0], dtype=np.float32)
            x2_vec = np.arange(roi_tcg2.shape[1], dtype=np.float32)
            ncx = bx1 + np.dot(np.sum(roi_tcg2, axis=0), x2_vec) / s2
            ncy = by1 + np.dot(np.sum(roi_tcg2, axis=1), y2_vec) / s2

            # Duplicate check against grayscale centroids (KDTree, O(log N))
            candidate = np.array([[ncx, ncy]], dtype=np.float32)
            is_new = True
            if gray_tree is not None:
                dist, _ = gray_tree.query(candidate, k=1)
                if dist[0] < MIN_SEPARATION_PX:
                    is_new = False

            # Duplicate check against previously accepted color centroids (linear scan)
            if is_new and len(color_pts_for_tree) > 0:
                color_arr = np.array(color_pts_for_tree, dtype=np.float32)
                dists = np.hypot(color_arr[:, 0] - ncx, color_arr[:, 1] - ncy)
                if np.min(dists) < MIN_SEPARATION_PX:
                    is_new = False

            if not is_new:
                continue

            new_color_centroids.append((ncx, ncy, ch_idx, max(4, int(max(w2, h2) * 0.5 + pad2))))
            color_pts_for_tree.append([ncx, ncy])

            # Draw per-channel colored marker
            col      = CHANNEL_COLORS[ch_idx]
            ncx_i, ncy_i = int(round(ncx)), int(round(ncy))
            vh       = max(4, int(max(w2, h2) * 0.5 + pad2))
            cv2.rectangle(debug_vis,
                          (max(0, ncx_i - vh), max(0, ncy_i - vh)),
                          (min(image.shape[1], ncx_i + vh), min(image.shape[0], ncy_i + vh)), col, 1)
            cv2.rectangle(raw_image_color,
                          (max(0, ncx_i - vh), max(0, ncy_i - vh)),
                          (min(raw_image_color.shape[1], ncx_i + vh), min(raw_image_color.shape[0], ncy_i + vh)), col, 1)
            cv2.drawMarker(raw_image_color, (ncx_i, ncy_i), color=col,
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)

    # Merge all detected centroids
    total_color  = len(new_color_centroids)
    all_centroids = centroids + [(x, y) for x, y, _, _ in new_color_centroids]

    # =========================================================================
    # UNIFIED VISUALIZATION (all detections on clean original image)
    # =========================================================================
    # Reloads the original image to draw detections without any intermediate
    # processing artifacts. All objects (grayscale + color-shifted) share the
    # same marker style: green bounding box + red crosshair.
    vis5 = cv2.imread(str(img_path))
    if vis5 is None:
        vis5 = raw_image_color.copy()

    BOX_COLOR  = (0, 255, 0)   # Green — all detected objects
    MARK_COLOR = (0, 0, 255)   # Red   — centroid crosshair
    BOX_HALF   = 8             # Fixed 16×16 px bounding box for readability

    for cx_f, cy_f in all_centroids:
        ci, cj = int(round(cx_f)), int(round(cy_f))
        vx1b = max(0, ci - BOX_HALF)
        vx2b = min(vis5.shape[1] - 1, ci + BOX_HALF)
        vy1b = max(0, cj - BOX_HALF)
        vy2b = min(vis5.shape[0] - 1, cj + BOX_HALF)
        cv2.rectangle(vis5, (vx1b, vy1b), (vx2b, vy2b), BOX_COLOR, 1)
        cv2.drawMarker(vis5, (ci, cj), color=MARK_COLOR,
                       markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    end_time       = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    n_red   = sum(1 for _, _, c, _ in new_color_centroids if c == 2)
    n_green = sum(1 for _, _, c, _ in new_color_centroids if c == 1)
    n_blue  = sum(1 for _, _, c, _ in new_color_centroids if c == 0)

    print(f"  [Stage 1] Isolated stars (grayscale)  : {len(centroids)}")
    print(f"  [Stage 2] Color-shifted objects        : +{total_color}  "
          f"(R: {n_red}  G: {n_green}  B: {n_blue})")
    print(f"  [Total  ] Unique celestial objects     : {len(all_centroids)}")
    print(f"  [Timing ] Inference time               : {inference_time:.2f} ms")

    # =========================================================================
    # SAVE OUTPUT IMAGES
    # =========================================================================
    cv2.imwrite(str(img_adjust_dir / "1_clean.png"),            clean)
    cv2.imwrite(str(img_adjust_dir / "2_binary.png"),           binary)
    cv2.imwrite(str(img_adjust_dir / "3_debug_centroids.png"),  debug_vis)
    cv2.imwrite(str(img_adjust_dir / "4_human_verify.png"),     raw_image_color)
    cv2.imwrite(str(img_adjust_dir / "5_visualization.png"),    vis5)

    print(f"  [Saved  ] → {img_adjust_dir}")
    print(f"             1_clean.png  |  2_binary.png  |  3_debug_centroids.png")
    print(f"             4_human_verify.png  |  5_visualization.png")

    # Visualization on screen
    plt.figure(figsize=(10, 6))
    plt.imshow(debug_vis)
    plt.title(f"Star Centroids — {img_path.name}")
    plt.axis("off")
    plt.show()

print(f"\n{'='*60}")
print(f"  All images processed successfully.")
print(f"{'='*60}\n")
