"""Blind development ablation; reference metadata is used ONLY after inference.

Run: python -m examples.check_image03_experimental
This is not a held-out benchmark, nor a full WCS attitude validation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from models.pipeline import PipelineConfig, StarTrackerPipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/image03-experimental")
    parser.add_argument("--noise-trials", type=int, default=20)
    args = parser.parse_args()
    if args.noise_trials < 0:
        parser.error("noise-trials must be nonnegative")
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = PipelineConfig.from_json(ROOT / "configs/examples/image03.json")
    calibrated = replace(baseline, fov_deg=16.8, fov_max_error_deg=.8)
    experiment = PipelineConfig.from_json(ROOT / "configs/examples/image03-experimental.json")
    pipeline = StarTrackerPipeline(baseline)  # One database load; no reference lookup.
    records = []
    configs = [
        ("historical_preset", baseline),
        ("calibration_only", calibrated),
        ("tight_gate_only", replace(calibrated, match_radius=.003)),
        ("wide_catalog_only", replace(calibrated, verification_catalog_stars=90)),
        ("gate_and_catalog_top20", replace(calibrated, match_radius=.003, verification_catalog_stars=90)),
        ("extended_candidates_only", replace(calibrated, max_stars=40, max_input_stars=40)),
        ("experimental_top40", experiment),
    ]
    image_path = ROOT / "dataset/03.bmp"
    for name, config in configs:
        pipeline.config = config
        result = pipeline.run(image_path)
        records.append({"case": name, "result": result})
        print(name, result["status"], result["num_matches"], flush=True)

    # Regression controls: old and opt-in verifier, same detector and FOV.
    stellarium = PipelineConfig.from_json(ROOT / "configs/examples/stellarium.json")
    stellarium_path = ROOT / "dataset/stellarium_test/stellarium_view.png"
    for name, config in (("stellarium_baseline", stellarium),
                         ("stellarium_opt_in", replace(stellarium, max_input_stars=40,
                          verification_catalog_stars=90, match_radius=.003))):
        pipeline.config = config
        result = pipeline.run(stellarium_path)
        records.append({"case": name, "result": result})
        print(name, result["status"], result["num_matches"], flush=True)

    fixture_root = ROOT.parent / "lost-evals/scenarios-pyramid/20-low-noise/images"
    for index in range(5):
        path = fixture_root / f"{index}.png"
        if not path.exists():
            continue
        for mode in ("baseline", "opt_in"):
            config = PipelineConfig(database=baseline.database, fov_deg=20, solve_timeout_ms=2000)
            if mode == "opt_in":
                config = replace(config, max_input_stars=40, verification_catalog_stars=90, match_radius=.003)
            pipeline.config = config
            records.append({"case": f"synthetic_{index}_{mode}", "result": pipeline.run(path)})

    # Negatives are controls, NOT an empirical false-positive-rate guarantee.
    rng = np.random.default_rng(20261025)
    pipeline.config = replace(experiment, solve_timeout_ms=2000)
    for index in range(args.noise_trials):
        image = np.clip(rng.normal(3, 1, (480, 640)), 0, 255).astype(np.uint8)
        # Unrelated Gaussian point sources, no catalog-derived geometry.
        yy, xx = np.indices(image.shape)
        signal = image.astype(float)
        for x, y, amplitude in zip(rng.uniform(15, 625, 20), rng.uniform(15, 465, 20), rng.uniform(10, 120, 20)):
            signal += amplitude * np.exp(-((xx-x)**2 + (yy-y)**2) / 2)
        records.append({"case": f"random_points_{index}",
                        "result": pipeline.run_array(np.clip(signal, 0, 255).astype(np.uint8))})
        print("negative", index, records[-1]["result"]["status"], flush=True)

    # References are read only AFTER every blind run above is finished.
    from examples.check_stellarium import reference, score
    labels = json.loads((ROOT / "dataset/stellarium_test/stars_labels.json").read_text())
    ref, _, _ = reference(labels)
    for record in records:
        result = record["result"]
        if record["case"].startswith("stellarium_"):
            record["metrics"] = score(result, labels, ref)
        elif record["case"] in dict(configs) and result["status"] == "ok":
            ra, dec = np.deg2rad([result["ra_deg"], result["dec_deg"]])
            ref_ra, ref_dec = np.deg2rad([48.773, 41.364])  # Rounded screenshot, NOT WCS.
            cosine = np.sin(dec)*np.sin(ref_dec) + np.cos(dec)*np.cos(ref_dec)*np.cos(ra-ref_ra)
            record["approx_boresight_error_deg"] = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    report = {
        "track": "DEVELOPMENT_ABLATION_NOT_HELD_OUT",
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "database_sha256": hashlib.sha256(baseline.database.read_bytes()).hexdigest(),
        "config_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "configs/examples").glob("*.json"))},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                        "negative_seed": 20261025},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "models").rglob("*.py"))},
        "records": records,
        "caveat": "FOV from user-provided Astrometry calibration; no center/roll/labels supplied to solver. "
                  "Rounded screenshot center supports approximate boresight comparison only, not full quaternion accuracy. "
                  "Timing is a single development run with soft timeouts, not a performance benchmark."
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = ["# Image03: thử nghiệm verifier và candidate selection", "",
            "Development trên ảnh đã biết, không phải held-out benchmark. Không đọc RA/Dec, roll hay nhãn trước inference.", "",
            "| Cấu hình | Trạng thái | Matches | RA / Dec (°) | Sai lệch tâm xấp xỉ (°) |",
            "|---|---|---:|---|---:|"]
    for record in records:
        if record["case"] not in dict(configs):
            continue
        result = record["result"]
        center = (f"{result['ra_deg']:.6f} / {result['dec_deg']:.6f}" if result["status"] == "ok" else "—")
        error = record.get("approx_boresight_error_deg")
        rows.append(f"| {record['case']} | {result['status']} | {result['num_matches']} | {center} | {f'{error:.6f}' if error is not None else '—'} |")
    rows += ["", "## Kiểm tra hồi quy", "",
             "| Stellarium | Matches | Cặp ID đúng / có nhãn | Sai số attitude (°) |",
             "|---|---:|---:|---:|"]
    for record in records:
        if record["case"].startswith("stellarium_"):
            metrics = record["metrics"]
            rows.append(f"| {record['case']} | {record['result']['num_matches']} | "
                        f"{metrics['pairwise_star_id_correct']}/{metrics['pairwise_star_id_labelled']} | "
                        f"{metrics['attitude_error_deg']:.6f} |")
    rows.append("")
    for mode in ("baseline", "opt_in"):
        subset = [r for r in records if r["case"].startswith("synthetic_") and r["case"].endswith(mode)]
        rows.append(f"- Synthetic smoke {mode}: {sum(r['result']['status']=='ok' for r in subset)}/{len(subset)} solve; chưa chấm ground truth ở runner này.")
    negatives = [r for r in records if r["case"].startswith("random_points_")]
    rows += [f"- Random-point negatives: {sum(r['result']['status']=='ok' for r in negatives)}/{len(negatives)} solve; mẫu nhỏ, không phải bảo đảm false-positive rate.", "",
             "## Diễn giải và giới hạn", "",
             "- Detector/centroid, threshold, Top-Hat và ranking flux của preset cũ được giữ nguyên. Không lọc theo nhãn/vị trí sao đã biết.",
             "- Input mở rộng 20 → 40; bỏ giới hạn ngầm 30 bằng max_input_stars=40 để sao rank32 được tham gia verification. Pattern search vẫn chỉ top16.",
             "- Verification pool tối đa90 sao catalog, thay vì cắt bằng số detection. Xác suất mismatch dùng đúng số sao thực trong pool mở rộng.",
             "- Radius 0.015 → 0.003: 9.6px → 1.92px tại width640; match_threshold vẫn0.001. Không nới threshold để ép nghiệm.",
             "- Các trường mới mặc định null giữ hành vi baseline. Preset cũ không bị ghi đè.",
             "- Tâm tham khảo48.773°/41.364° lấy từ screenshot Astrometry job17182488; chưa có WCS gốc để chấm roll/quaternion đầy đủ.",
             "- Không kết luận cải thiện DUST hoặc benchmark LOST đầy đủ từ smoke test này.", "",
             "## Chạy lại", "", "```bash", "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m examples.check_image03_experimental", "```", "",
             "Chi tiết config, source checksum và mọi kết quả: [results.json](results.json). Overlay chỉ vẽ ID thực sự do solver trả về."]
    (args.output_dir / "summary.md").write_text("\n".join(rows) + "\n")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.cvtColor(np.clip(image.astype(float)*8, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    result = next(r["result"] for r in records if r["case"] == "experimental_top40")
    for point, hip in zip(result.get("matched_centroids_yx", []), result.get("matched_catalog_ids") or []):
        y, x = np.rint(point).astype(int)
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), 1)
        cv2.putText(overlay, f"HIP {hip}", (x+9, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    if not cv2.imwrite(str(args.output_dir / "image03-overlay.png"), overlay):
        raise OSError("Could not write overlay")


if __name__ == "__main__":
    main()
