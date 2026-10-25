"""Windows-only Stellarium screenshot/label collection utility.

Optional collection dependencies: mss, pygetwindow, requests, astropy and
astroquery. This tool uses Stellarium API sky hints; it is NOT the blind USSEG
image pipeline. Run it separately on the Windows host with Stellarium open.
"""

import ctypes
import ctypes.wintypes
import json
import math
import time
import warnings
import os
import cv2
import mss
import numpy as np
import pygetwindow as gw
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from scipy.spatial import cKDTree

# 1. Bật DPI Awareness chuẩn 1920x1080 trên Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

STELLARIUM_URL = "http://localhost:8090/api"
MATCH_RADIUS_PX = 3.0

COMMON_STAR_NAMES = {
    677: "Alpheratz (alpha And)",
    3092: "delta And",
    3179: "Schedar (alpha Cas)",
    3031: "epsilon And",
    2912: "pi And",
    3881: "nu And",
    4436: "mu And",
    5447: "Mirach (beta And)",
    7588: "Achernar (alpha Eri)",
    9640: "Almach (gamma And)",
    9884: "Hamal (alpha Ari)",
    11767: "Polaris (alpha UMi)",
    14576: "Algol (beta Per)",
    15863: "Mirfak (alpha Per)",
    21421: "Aldebaran (alpha Tau)",
    24436: "Rigel (beta Ori)",
    24608: "Capella (alpha Aur)",
    25336: "Bellatrix (gamma Ori)",
    25930: "Mintaka (delta Ori)",
    26311: "Alnilam (epsilon Ori)",
    26727: "Alnitak (zeta Ori)",
    27989: "Betelgeuse (alpha Ori)",
    30438: "Canopus (alpha Car)",
    32349: "Sirius (alpha CMa)",
    36850: "Castor (alpha Gem)",
    37279: "Procyon (alpha CMi)",
    37826: "Pollux (beta Gem)",
    49669: "Regulus (alpha Leo)",
    54061: "Dubhe (alpha UMa)",
    65474: "Spica (alpha Vir)",
    69673: "Arcturus (alpha Boo)",
    80763: "Antares (alpha Sco)",
    85888: "pi Her",
    86414: "iota Her",
    87563: "90 Her",
    87808: "theta Her",
    88788: "xi Her",
    89826: "kappa Lyr",
    89981: "mu Lyr",
    90191: "mu Her",
    91262: "Vega (alpha Lyr)",
    91919: "epsilon1 Lyr",
    91926: "epsilon2 Lyr",
    91971: "zeta1 Lyr",
    91973: "zeta2 Lyr",
    92420: "Sheliak (beta Lyr)",
    92791: "delta2 Lyr",
    92728: "delta1 Lyr",
    92862: "R Lyr",
    93104: "nu Lyr",
    93408: "16 Lyr",
    93805: "lambda Aql",
    94481: "eta Lyr (Aladfar)",
    94713: "theta Lyr",
    95066: "20 Aql",
    95501: "delta Aql (Almizan I)",
    95585: "nu Aql",
    95947: "Albireo (beta1 Cyg)",
    96229: "mu Aql",
    96468: "eta Aql (Bezek)",
    96807: "sigma Aql",
    97165: "Fawaris (delta Cyg)",
    97278: "Tarazed (gamma Aql)",
    97649: "Altair (alpha Aql)",
    97675: "phi Aql",
    97804: "eta Aql",
    97938: "xi Aql (Libertas)",
    98036: "Alshain (beta Aql)",
    98110: "eta Cyg",
    99473: "Tseen Foo (theta Aql)",
    99675: "31 Cyg (Omicron1 Cyg)",
    100453: "Sadr (gamma Cyg)",
    101421: "epsilon Del (Aldulfin)",
    101769: "Rotanev (beta Del)",
    101958: "Sualocin (alpha Del)",
    102098: "Deneb (alpha Cyg)",
    102488: "Aljanah (epsilon Cyg)",
    113368: "Fomalhaut (alpha PsA)",
}


def api_request(path, data=None):
    method = requests.get if data is None else requests.post
    response = method(f"{STELLARIUM_URL}/{path}", timeout=10,
                      **({} if data is None else {"data": data}))
    response.raise_for_status()
    if response.text.strip().lower().startswith("error"):
        raise RuntimeError(f"Stellarium {path}: {response.text}")
    return response


def set_prop(prop_id, val):
    value = str(val).lower() if isinstance(val, bool) else str(val)
    api_request("stelproperty/set", {"id": prop_id, "value": value})


def freeze_and_clean_stellarium(saved):
    """Lưu trạng thái, dừng thời gian và xác nhận ảnh chụp dùng perspective."""
    print("[1/5] Đang đóng băng thời gian và ẩn giao diện Stellarium...")
    props = api_request("stelproperty/list").json()
    saved["time"] = api_request("main/status").json()["time"]
    targets = {
        "StelCore.currentProjectionTypeKey": "ProjectionPerspective",
        "StelGui.visible": False,
        "StarMgr.flagLabelsDisplayed": False,
        "SolarSystem.labelsDisplayed": False,
        "NebulaMgr.flagHintDisplayed": False,
        "ConstellationMgr.linesDisplayed": False,
        "ConstellationMgr.namesDisplayed": False,
        "ConstellationMgr.artDisplayed": False,
        "ConstellationMgr.boundariesDisplayed": False,
        "GridLinesMgr.gridlinesDisplayed": False,
        "LandscapeMgr.landscapeDisplayed": False,
        "LandscapeMgr.atmosphereDisplayed": False,
        "LandscapeMgr.fogDisplayed": False,
        "LandscapeMgr.cardinalPointsDisplayed": False,
    }
    if "Oculars.flagGuiPanelEnabled" in props:
        targets["Oculars.flagGuiPanelEnabled"] = False
    saved["props"] = {}
    api_request("main/time", {"timerate": 0})
    for key, value in targets.items():
        if key not in props or not props[key]["isWritable"]:
            raise RuntimeError(f"Stellarium không hỗ trợ property: {key}")
        saved["props"][key] = props[key]["value"]
        set_prop(key, value)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        actual = api_request("stelproperty/list").json()
        rate = api_request("main/status").json()["time"]["timerate"]
        if rate == 0 and all(actual[k]["value"] == v for k, v in targets.items()):
            time.sleep(1.0)  # Đợi hiệu ứng fade và khung hình mới.
            return
        time.sleep(0.1)
    raise RuntimeError("Stellarium chưa dừng thời gian/ẩn GUI/đổi perspective.")


def restore_stellarium_gui(saved):
    """Khôi phục trạng thái ban đầu kể cả khi chụp thất bại."""
    errors = []
    for key, value in reversed(list(saved.get("props", {}).items())):
        try:
            set_prop(key, value)
        except Exception as exc:
            errors.append(str(exc))
    if "time" in saved:
        try:
            api_request("main/time", {"timerate": saved["time"]["timerate"]})
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        warnings.warn("Không khôi phục được hết trạng thái: " + "; ".join(errors))


def focus_and_capture_stellarium():
    def is_stellarium_window(win):
        pid = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(win._hWnd, ctypes.byref(pid))
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return False
        try:
            path = ctypes.create_unicode_buffer(32768)
            size = ctypes.wintypes.DWORD(len(path))
            kernel.QueryFullProcessImageNameW.argtypes = [ctypes.wintypes.HANDLE,
                ctypes.wintypes.DWORD, ctypes.wintypes.LPWSTR,
                ctypes.POINTER(ctypes.wintypes.DWORD)]
            return bool(kernel.QueryFullProcessImageNameW(handle, 0, path,
                ctypes.byref(size))) and os.path.basename(path.value).lower() == "stellarium.exe"
        finally:
            kernel.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
            kernel.CloseHandle(handle)

    windows = [win for win in gw.getAllWindows()
               if win.visible and is_stellarium_window(win)]
    if not windows:
        raise RuntimeError(
            "Không tìm thấy cửa sổ Stellarium! Hãy mở Stellarium lên trước."
        )

    win = windows[0]
    if win.isMinimized:
        win.restore()

    try:
        win.activate()
    except Exception:
        # Windows có thể chặn SetForegroundWindow từ tiến trình nền.
        user = ctypes.windll.user32
        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        foreground_thread = user.GetWindowThreadProcessId(user.GetForegroundWindow(), None)
        attached = user.AttachThreadInput(current_thread, foreground_thread, True)
        try:
            user.BringWindowToTop(win._hWnd)
            user.SetForegroundWindow(win._hWnd)
        finally:
            if attached:
                user.AttachThreadInput(current_thread, foreground_thread, False)

    time.sleep(0.5)
    if ctypes.windll.user32.GetForegroundWindow() != win._hWnd:
        raise RuntimeError("Stellarium chưa ở phía trước; không chụp màn hình ứng dụng khác.")

    # Chỉ chụp client area, bỏ title bar và viền Windows.
    rect = ctypes.wintypes.RECT()
    origin = ctypes.wintypes.POINT(0, 0)
    if not ctypes.windll.user32.GetClientRect(win._hWnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    if not ctypes.windll.user32.ClientToScreen(win._hWnd, ctypes.byref(origin)):
        raise ctypes.WinError()
    monitor = {"top": origin.y, "left": origin.x,
               "width": rect.right, "height": rect.bottom}

    with mss.MSS() as sct:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def get_stellarium_camera_params():
    status_resp = api_request("main/status").json()
    fov_deg = float(status_resp["view"]["fov"])

    view_resp = api_request("main/view").json()
    j2000_raw = view_resp["j2000"]
    vec = json.loads(j2000_raw) if isinstance(j2000_raw, str) else j2000_raw

    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm > 0:
        vx, vy, vz = vx / norm, vy / norm, vz / norm

    ra_rad = math.atan2(vy, vx)
    if ra_rad < 0:
        ra_rad += 2 * math.pi
    dec_rad = math.asin(max(min(vz, 1.0), -1.0))

    return math.degrees(ra_rad), math.degrees(dec_rad), fov_deg


def query_catalog_stars(ra_deg, dec_deg, radius_deg, mag_limit=6.0):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    v_hip = Vizier(
        columns=["HIP", "Vmag", "_RAJ2000", "_DEJ2000"],
        column_filters={"Vmag": f"<{mag_limit}"},
    )
    v_hip.ROW_LIMIT = 2500
    res_hip = v_hip.query_region(
        coord, radius=radius_deg * u.deg, catalog="I/239/hip_main"
    )
    if not res_hip or len(res_hip[0]) == 0:
        return []

    hip_to_name = dict(COMMON_STAR_NAMES)
    try:
        v_names = Vizier(columns=["HIP", "Name", "Bayer", "Fl", "Cst"])
        v_names.ROW_LIMIT = 2500
        res_names = v_names.query_region(
            coord, radius=radius_deg * u.deg, catalog="IV/27A/catalog"
        )
        if res_names and len(res_names[0]) > 0:
            for row in res_names[0]:
                if np.ma.is_masked(row["HIP"]):
                    continue
                h_id = int(row["HIP"])
                if h_id in hip_to_name:
                    continue

                proper = (
                    str(row["Name"]).strip()
                    if "Name" in row.colnames and not np.ma.is_masked(row["Name"])
                    else ""
                )
                bayer = (
                    str(row["Bayer"]).strip()
                    if "Bayer" in row.colnames
                    and not np.ma.is_masked(row["Bayer"])
                    else ""
                )
                flam = (
                    str(row["Fl"]).strip()
                    if "Fl" in row.colnames and not np.ma.is_masked(row["Fl"])
                    else ""
                )
                cst = (
                    str(row["Cst"]).strip()
                    if "Cst" in row.colnames and not np.ma.is_masked(row["Cst"])
                    else ""
                )

                desig = ""
                if bayer and cst:
                    desig = f"{bayer} {cst}"
                elif flam and cst:
                    desig = f"{flam} {cst}"

                if proper and desig:
                    hip_to_name[h_id] = f"{proper} ({desig})"
                elif proper:
                    hip_to_name[h_id] = proper
                elif desig:
                    hip_to_name[h_id] = desig
    except Exception:
        pass

    table = res_hip[0]
    stars = []
    for row in table:
        try:
            ra_val = row["_RAJ2000"]
            dec_val = row["_DEJ2000"]
            vmag_val = row["Vmag"]

            if (
                np.ma.is_masked(ra_val)
                or np.ma.is_masked(dec_val)
                or np.ma.is_masked(vmag_val)
            ):
                continue

            hip_id = int(row["HIP"])
            star_name = hip_to_name.get(hip_id, f"HIP {hip_id}")

            stars.append({
                "hip": hip_id,
                "name": star_name,
                "vmag": float(vmag_val),
                "ra": float(ra_val),
                "dec": float(dec_val),
            })
        except Exception:
            continue

    stars.sort(key=lambda x: x["vmag"])
    return stars


def detect_star_blobs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    gray_masked = gray  # GUI đã được ẩn; giữ cả sao gần mép ảnh.

    _, thresh = cv2.threshold(gray_masked, 140, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        if 2 <= area <= 900:
            r = max(4, int(math.sqrt(area) * 1.5))
            y1, y2 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
            x1, x2 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
            flux = float(np.sum(gray_masked[y1:y2, x1:x2]))
            blobs.append((cx, cy, flux))

    blobs.sort(key=lambda x: x[2], reverse=True)
    return blobs


def compute_unrotated_offsets(stars, ra0, dec0, fov_deg, h):
    f_px = (h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    a0 = math.radians(ra0)
    d0 = math.radians(dec0)

    cat_uv = []
    for s in stars:
        a = math.radians(s["ra"])
        d = math.radians(s["dec"])
        cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(
            a - a0
        )
        if cos_c <= 0.05:
            continue
        xi = (math.cos(d) * math.sin(a - a0)) / cos_c
        eta = (
            math.sin(d) * math.cos(d0)
            - math.cos(d) * math.sin(d0) * math.cos(a - a0)
        ) / cos_c

        u_x = -xi * f_px
        u_y = -eta * f_px
        cat_uv.append((u_x, u_y, s))

    return cat_uv


def solve_global_alignment(cat_uv, blobs, w, h):
    """
    Khớp toàn cục đa điểm (Global Alignment):
    Quét góc xoay 1D trước, sau đó dùng RANSAC / Affine tối ưu hóa toàn bộ
    Chỉ phù hợp khi ảnh và catalog dùng cùng phép chiếu perspective.
    """
    cx, cy = w / 2.0, h / 2.0
    if len(blobs) < 5 or len(cat_uv) < 5:
        raise RuntimeError("Cần ít nhất 5 sao catalog và 5 blob để căn chỉnh.")

    top_blobs = np.array([[b[0], b[1]] for b in blobs[:80]])
    tree = cKDTree(top_blobs)

    best_score = -1
    best_theta = 0.0

    for step in range(0, 3600):
        theta = math.radians(step * 0.1)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        pts = []
        for u_x, u_y, _ in cat_uv[:40]:
            px = cx + (u_x * cos_t - u_y * sin_t)
            py = cy + (u_x * sin_t + u_y * cos_t)
            if 10 <= px < (w - 10) and 10 <= py < (h - 10):
                pts.append([px, py])
        if not pts:
            continue
        dists, _ = tree.query(pts)
        score = np.sum(dists < 14.0)
        if score > best_score:
            best_score = score
            best_theta = theta

    cos_t, sin_t = math.cos(best_theta), math.sin(best_theta)
    all_blobs = np.array([[b[0], b[1]] for b in blobs])
    all_tree = cKDTree(all_blobs)

    src_pts, dst_pts = [], []
    for u_x, u_y, _ in cat_uv:
        px = cx + (u_x * cos_t - u_y * sin_t)
        py = cy + (u_x * sin_t + u_y * cos_t)
        if 10 <= px < (w - 10) and 10 <= py < (h - 10):
            dist, idx = all_tree.query([px, py])
            if dist < 18.0:  # Nới rộng biên độ khớp cho vùng rìa
                src_pts.append([u_x, u_y])
                dst_pts.append(all_blobs[idx])

    if len(src_pts) >= 5:
        # Dùng RANSAC với threshold 3px để loại bỏ nhiễu biên và giữ lại các điểm chuẩn tuyệt đối
        M, inliers = cv2.estimateAffinePartial2D(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if M is not None and int(inliers.sum()) >= 5:
            return M

    raise RuntimeError("Không tìm được alignment tin cậy; không xuất label.")


def main():
    saved = {}
    try:
        freeze_and_clean_stellarium(saved)
        print("[2/5] Đang chụp vùng ảnh Stellarium...")
        img = focus_and_capture_stellarium()
        h, w = img.shape[:2]
        ra0, dec0, fov = get_stellarium_camera_params()
    finally:
        restore_stellarium_gui(saved)

    print(f"      + Kích thước vật lý: {w}x{h} | FOV dọc: {fov:.4f}°")
    print(f"      + Tâm J2000: RA={ra0:.5f}°, Dec={dec0:.5f}°")

    diag_fov = math.degrees(math.atan(math.tan(math.radians(fov / 2))
                                       * math.hypot(1, w / h))) + 0.5
    print(f"[3/5] Đang tải danh mục sao Hipparcos (bán kính {diag_fov:.2f}°)...")
    stars = query_catalog_stars(ra0, dec0, radius_deg=diag_fov, mag_limit=6.0)
    print(f"      + Tải về {len(stars)} sao từ catalog.")

    print(
        "[4/5] Đang tối ưu hóa định vị toàn cục (Global Alignment & RANSAC)..."
    )
    blobs = detect_star_blobs(img)
    cat_uv = compute_unrotated_offsets(stars, ra0, dec0, fov, h)
    M = solve_global_alignment(cat_uv, blobs, w, h)

    roll_deg = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    print(f"      + Góc xoay khung nhìn chuẩn xác: {roll_deg:.2f}°")

    print("[5/5] Đang xuất JSON và tạo 2 ảnh Debug (HIP & Tên Sao)...")
    all_blobs = (
        np.array([[b[0], b[1]] for b in blobs]) if blobs else np.empty((0, 2))
    )
    all_tree = cKDTree(all_blobs) if len(all_blobs) > 0 else None

    labeled_stars = []
    rejected_stars = []
    used_blobs = set()
    overlay_hip_img = img.copy()
    overlay_name_img = img.copy()

    for u_x, u_y, s in cat_uv:
        px = float(M[0, 0] * u_x + M[0, 1] * u_y + M[0, 2])
        py = float(M[1, 0] * u_x + M[1, 1] * u_y + M[1, 2])

        if not (8 <= px < w - 8 and 8 <= py < h - 8):
            continue
        dist, idx = all_tree.query([px, py])
        reason = None
        if dist > MATCH_RADIUS_PX:
            reason = "no_blob_within_threshold"
        elif int(idx) in used_blobs:
            reason = "blob_already_assigned"
        if reason:
            rejected_stars.append({"hip_id": s["hip"], "reason": reason,
                                   "predicted_x": round(px, 2),
                                   "predicted_y": round(py, 2),
                                   "residual_px": round(float(dist), 3)})
            continue
        used_blobs.add(int(idx))
        predicted_x, predicted_y = px, py
        px, py = map(float, all_blobs[idx])

        if 8 <= px < w - 8 and 8 <= py < h - 8:
            labeled_stars.append({
                "predicted_x": round(predicted_x, 2),
                "predicted_y": round(predicted_y, 2),
                "residual_px": round(float(dist), 3),
                "match_status": "matched",
                "hip_id": s["hip"],
                "name": s["name"],
                "vmag": round(s["vmag"], 2),
                "x": round(px, 2),
                "y": round(py, 2),
                "ra_deg": round(s["ra"], 5),
                "dec_deg": round(s["dec"], 5),
            })

            ix, iy = int(round(px)), int(round(py))

            # 1. Ảnh debug theo mã HIP
            cv2.circle(overlay_hip_img, (ix, iy), 9, (0, 255, 0), 1)
            cv2.putText(
                overlay_hip_img,
                f"HIP {s['hip']}",
                (ix + 11, iy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
            )

            # 2. Ảnh debug theo Tên Sao
            is_named = not s["name"].startswith("HIP ")
            circle_color = (0, 200, 255) if is_named else (0, 255, 0)
            text_color = (100, 255, 255) if is_named else (180, 220, 180)
            font_scale = 0.48 if is_named else 0.38

            cv2.circle(overlay_name_img, (ix, iy), 9, circle_color, 1)
            cv2.putText(
                overlay_name_img,
                s["name"],
                (ix + 11, iy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
            )

    output_json = {
        "projection": "ProjectionPerspective",
        "match_radius_px": MATCH_RADIUS_PX,
        "alignment_matrix": M.tolist(),
        "rejected_stars": rejected_stars,
        "fov_deg": round(fov, 4),
        "center_ra_deg": round(ra0, 5),
        "center_dec_deg": round(dec0, 5),
        "roll_angle_deg": round(roll_deg, 3),
        "image_width": w,
        "image_height": h,
        "total_stars": len(labeled_stars),
        "stars": labeled_stars,
    }

    print(f"      + Giữ {len(labeled_stars)} label; loại {len(rejected_stars)} ứng viên.")

    cv2.imwrite("stellarium_view.png", img)
    with open("stars_labels.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    cv2.imwrite("stellarium_debug_hip.png", overlay_hip_img)
    cv2.imwrite("stellarium_debug_names.png", overlay_name_img)

    print(f"\n[✓] Hoàn tất! Đã lưu {len(labeled_stars)} sao vào 'stars_labels.json'.")
    print("[✓] Ảnh 1 (Theo mã HIP):  'stellarium_debug_hip.png'")
    print("[✓] Ảnh 2 (Theo Tên Sao): 'stellarium_debug_names.png'")


if __name__ == "__main__":
    main()
