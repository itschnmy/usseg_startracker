"""Publish real example images, run file runtime, then score/visualize results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from usseg_runtime.config import ROOT
from usseg_runtime.engine import Runner
from usseg_runtime.publish import publish_encoded
from usseg_runtime.storage import atomic_bytes, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; will not overwrite")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "input"
    inputs.mkdir()
    sources = [
        ("image03", ROOT/"dataset/03.bmp", "lab640"),
        ("image200701-raw", ROOT/"dataset/200701.bmp", "lab640"),
        ("image200701-calibrated", ROOT/"dataset/calibration/openmv/session_01/200701_calibrated.bmp", "lab640"),
        ("stellarium", ROOT/"dataset/stellarium_test/stellarium_view.png", "stellarium1920"),
    ]
    for job, source, profile in sources:
        publish_encoded(source, inputs, job, profile)
    runner = Runner(inputs, args.output/"output", ROOT/"configs/runtime/development.json")
    code = runner.run()
    report = {"exit_code": code, "run_id": runner.run_id, "track": "development_runtime_smoke",
              "notes": "Input pixels as supplied; no extra dark subtraction. Lab FOV, no independent camera WCS.",
              "records": runner.current}
    rows = ["# Version3 camera runtime smoke", "", "Known development images; original/calibrated pixels used as supplied, no additional dark subtraction.", "",
            "| Job | Status | Matches | RA / Dec | HFOV | Fit RMSE arcsec | Worker ms |",
            "|---|---|---:|---|---:|---:|---:|"]
    by_job = {row["job_id"]: row for row in runner.current}
    for job, source, _ in sources:
        record = by_job[job]
        result = record.get("algorithm", {})
        center = f"{result['ra_deg']:.6f} / {result['dec_deg']:.6f}" if record["status"] == "solved" else "—"
        fov = f"{result['solved_fov_deg']:.6f}" if record["status"] == "solved" else "—"
        rmse = f"{result['rmse_arcsec']:.3f}" if record["status"] == "solved" else "—"
        rows.append(f"| {job} | {record['status']} | {result.get('num_matches',0)} | {center} | {fov} | {rmse} | {record.get('worker_total_ns',0)/1e6:.3f} |")
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        gain = 1 if job == "stellarium" else 8
        overlay = cv2.cvtColor(np.clip(gray.astype(float)*gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for point, hip in zip(result.get("matched_centroids_yx", []), result.get("matched_catalog_ids") or []):
            y, x = np.rint(point).astype(int)
            cv2.circle(overlay, (x, y), 8, (0,255,0), 1)
            cv2.putText(overlay, f"HIP {hip}", (max(0,min(x+9,gray.shape[1]-100)),y), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
        success, data = cv2.imencode(".png", overlay)
        if not success:
            raise OSError("Overlay encode failed")
        atomic_bytes(args.output/f"{job}-result.png", data.tobytes())
        rows.extend(["", f"![{job} solver result]({job}-result.png)", ""])
    # Reference is loaded after every blind inference has completed.
    from examples.check_stellarium import reference, score
    labels = json.loads((ROOT/"dataset/stellarium_test/stars_labels.json").read_text())
    ref, _, _ = reference(labels)
    report["stellarium_metrics"] = score(by_job["stellarium"]["algorithm"], labels, ref)
    report["summary"] = json.loads((runner.run_dir/"summary.json").read_text())
    rows += ["", f"Stellarium attitude error: {report['stellarium_metrics']['attitude_error_deg']} deg.",
             "03/200701 full attitude accuracy unavailable without their original WCS. Fit RMSE is not independent attitude error.",
             "Timing is one host development run; no flight-hardware throughput claim."]
    atomic_json(args.output/"report.json", report)
    atomic_bytes(args.output/"summary.md", ("\n".join(rows)+"\n").encode())
    print(json.dumps({"output": str(args.output), "exit_code": code, "statuses": report["summary"]["statuses_this_run"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
