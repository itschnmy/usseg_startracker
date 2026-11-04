"""Small release acceptance smoke test. Full pytest remains a ground-side test."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .config import ROOT
from .engine import Runner
from .publish import publish_encoded
from .storage import atomic_bytes, atomic_json


def self_test(output_dir: Path, config_path: Path):
    # Use a unique, durable run directory; never modify flight input.
    from .engine import utc_now
    import uuid
    directory = output_dir / "self-test" / uuid.uuid4().hex
    input_dir = directory / "input"
    input_dir.mkdir(parents=True)
    checks = []
    cases = [("image03", ROOT/"dataset/03.bmp", "lab640"),
             ("stellarium", ROOT/"dataset/stellarium_test/stellarium_view.png", "stellarium1920")]
    for name, image, profile in cases:
        publish_encoded(image, input_dir, name, profile)
    success, black = cv2.imencode(".png", np.zeros((480, 640), np.uint8))
    if not success:
        raise RuntimeError("Could not encode self-test black fixture")
    black_path = directory / "black.png"
    atomic_bytes(black_path, black.tobytes())
    publish_encoded(black_path, input_dir, "black", "lab640")
    # A corrupt payload with a valid marker must be isolated to that frame.
    bad = publish_encoded(ROOT/"dataset/03.bmp", input_dir, "corrupt", "lab640")
    atomic_bytes(bad/"frame.bmp", b"corrupt")
    runner = Runner(input_dir, directory/"output", config_path)
    exit_code = runner.run()
    records = {row["job_id"]: row for row in runner.current}
    checks.append(("batch_error_exit", exit_code == 3, f"exit={exit_code}"))
    for name, matches in (("image03", 7), ("stellarium", 16)):
        row = records.get(name, {})
        checks.append((name, row.get("status") == "solved" and row.get("algorithm", {}).get("num_matches") == matches,
                       f"status={row.get('status')}, matches={row.get('algorithm', {}).get('num_matches')}"))
    checks.append(("black_no_solve", records.get("black", {}).get("status") == "no_solve", "black frame"))
    checks.append(("corrupt_rejected", records.get("corrupt", {}).get("status") == "invalid_input", "bad payload"))
    # Check numerical attitude against independent simulator label metadata after inference.
    from examples.check_stellarium import reference, score
    labels = json.loads((ROOT/"dataset/stellarium_test/stars_labels.json").read_text())
    ref, _, _ = reference(labels)
    result = records.get("stellarium", {}).get("algorithm")
    if result is not None:
        metrics = score(result, labels, ref)
        error = metrics["attitude_error_deg"]
        checks.append(("stellarium_reference", error is not None and error < .01, f"attitude_error_deg={error}"))
    else:
        checks.append(("stellarium_reference", False, "No algorithm result"))
    if records.get("image03", {}).get("pose"):
        q = np.array(records["image03"]["pose"]["quaternion_wxyz"])
        checks.append(("quaternion_norm", abs(np.linalg.norm(q)-1) < 1e-9, "unit quaternion"))
    before = len(list((directory/"output/runs").glob("*/results/*.json")))
    resume = Runner(input_dir, directory/"output", config_path)
    resumed_code = resume.run()
    after = len(list((directory/"output/runs").glob("*/results/*.json")))
    checks.append(("resume_no_duplicates", before == after == 4 and not resume.current,
                   f"results_before={before}, after={after}, exit={resumed_code}"))
    failed = sum(not ok for _, ok, _ in checks)
    report = {"schema_version": "1.0", "completed_at_utc": utc_now(), "passed": len(checks)-failed,
              "failed": failed, "checks": [{"name": name, "passed": bool(ok), "detail": detail}
                                           for name, ok, detail in checks]}
    atomic_json(directory/"report.json", report)
    suite = ET.Element("testsuite", name="usseg-release-self-test", tests=str(len(checks)), failures=str(failed))
    for name, ok, detail in checks:
        case = ET.SubElement(suite, "testcase", name=name)
        if not ok:
            ET.SubElement(case, "failure", message=detail)
    atomic_bytes(directory/"junit.xml", ET.tostring(suite, encoding="utf-8", xml_declaration=True))
    print(json.dumps({"report": str(directory/"report.json"), "passed": report["passed"], "failed": failed}))
    return 0 if not failed else 3
