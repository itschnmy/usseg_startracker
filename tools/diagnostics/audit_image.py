"""Export detector evidence for an image; no labels are used for solving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from models.pipeline import PipelineConfig, StarTrackerPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--fov", required=True, type=float, help="Horizontal FOV in degrees")
    parser.add_argument(
        "--solve-timeout-ms", type=float, default=PipelineConfig.solve_timeout_ms,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        database=args.database, fov_deg=args.fov,
        solve_timeout_ms=args.solve_timeout_ms,
    )
    # A separate diagnostic detection pass exports all components. Its time is
    # deliberately excluded from the pipeline timing reported below.
    pipeline = StarTrackerPipeline(config)
    stars = pipeline.detect_stars(image, limit=False)
    # Display stretch only. Detector and solver receive the original uint8.
    display = cv2.cvtColor(np.clip(image.astype(float) * 8, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    selected = stars[:config.max_stars]
    for n, star in enumerate(selected):
        x, y = np.rint(star.position).astype(int)
        cv2.circle(display, (x, y), 7, (0, 255, 0), 1)
        cv2.putText(display, str(n + 1), (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    overlay_path = args.output_dir / f"{args.image.stem}-centroids.png"
    if not cv2.imwrite(str(overlay_path), display):
        raise OSError(f"Cannot write {overlay_path}")
    result = pipeline.run_array(image, source=str(args.image.resolve()))
    data = {
        "image": str(args.image.resolve()), "horizontal_fov_assumption_deg": args.fov,
        "has_independent_attitude_reference": False,
        "display_gain_only": 8, "detector_components_before_top20": len(stars),
        "percentiles": dict(zip(["min", "p50", "p99", "max"], np.percentile(image, [0, 50, 99, 100]).tolist())),
        "top20": [{"xy": s.position.tolist(), "peak": s.peak, "intensity": s.intensity, "radius": s.radius} for s in selected],
        "pipeline": result,
    }
    (args.output_dir / f"{args.image.stem}-audit.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
