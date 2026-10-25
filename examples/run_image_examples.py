"""Run image-specific presets blind; only then score available Stellarium labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from models.pipeline import PipelineConfig, StarTrackerPipeline
from examples.check_stellarium import reference, score


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"out/image-examples")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, image_path in (("image03", ROOT/"dataset/03.bmp"),
                              ("stellarium", ROOT/"dataset/stellarium_test/stellarium_view.png")):
        config_path = ROOT/"configs/examples"/f"{name}.json"
        config = PipelineConfig.from_json(config_path)
        result = StarTrackerPipeline(config).run(image_path)
        record = {"case": name, "config_file": str(config_path), "result": result,
                  "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((ROOT/"models").rglob("*.py"))}}
        records.append(record)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gain = 8 if name == "image03" else 1  # Display only; NEVER solver input.
        overlay = cv2.cvtColor(np.clip(image.astype(float)*gain,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for rank, point in enumerate(result["centroids_xy"], 1):
            x, y = np.rint(point).astype(int)
            cv2.circle(overlay, (x, y), 8, (0, 255, 255), 1)
            cv2.putText(overlay, str(rank), (x+9, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
        # Only IDs returned by the actual solver are drawn, not label IDs.
        for point, catalog_id in zip(result.get("matched_centroids_yx", []), result.get("matched_catalog_ids") or []):
            y, x = np.rint(point).astype(int)
            cv2.circle(overlay, (x, y), 11, (0, 255, 0), 1)
            cv2.putText(overlay, f"HIP {catalog_id}", (x+12, y+14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        cv2.putText(overlay, f"{result['status']} | matches={result['num_matches']}",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 1)
        if not cv2.imwrite(str(args.output_dir/f"{name}-overlay.png"), overlay):
            raise OSError("Failed to write diagnostic image")
        print(name, result["status"], result["num_matches"], flush=True)

    # Read labels AFTER all blind runs. No oracle substitution for image03.
    labels = json.loads((ROOT/"dataset/stellarium_test/stars_labels.json").read_text())
    ref, _, _ = reference(labels)
    records[1]["metrics"] = score(records[1]["result"], labels, ref)
    records[0]["reference_status"] = "unavailable: historical manual annotation is not verified ground truth"
    for record in records:
        (args.output_dir/f"{record['case']}.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = ["# Configured USSEG image examples", "",
            "Development examples tuned on these images, NOT a held-out benchmark. Original benchmark defaults remain unchanged.", "",
            "| Case | Status | Matches | Attitude error vs labels |",
            "|---|---|---:|---:|"]
    for record in records:
        error = record.get("metrics", {}).get("attitude_error_deg")
        rows.append(f"| {record['case']} | {record['result']['status']} | {record['result']['num_matches']} | {f'{error:.6f} deg' if error is not None else 'unavailable'} |")
    rows.extend(["", "FOV is horizontal. Stellarium label VFOV 16.9761 deg becomes HFOV 29.717817 deg.",
                 "Image03 preset FOV is an exploratory assumption, NOT a measured calibration. A no_solve is retained honestly.",
                 "Overlays: yellow = selected detections, green HIP IDs = actual solver matches. Image03 display gain 8x does not affect solving.",
                 "Stellarium labels are a pseudo-reference. These per-image presets must not be reported as an unbiased DUST/LOST-evals improvement."])
    (args.output_dir/"summary.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
