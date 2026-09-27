from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import PipelineConfig, StarTrackerPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified USSEG star-tracker pipeline")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--fov", required=True, type=float)
    parser.add_argument("--fov-max-error", type=float, default=5.0)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-stars", type=int, default=20)
    parser.add_argument("--pattern-checking-stars", type=int, default=12)
    parser.add_argument("--match-radius", type=float, default=0.01)
    parser.add_argument("--match-threshold", type=float, default=1e-3)
    parser.add_argument("--solve-timeout-ms", type=float, default=5000.0)
    parser.add_argument(
        "--attitude-method",
        choices=("SVD", "QUEST", "DAVENPORT", "TRIAD"),
        default="SVD",
    )
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        database=args.database,
        fov_deg=args.fov,
        fov_max_error_deg=args.fov_max_error,
        sigma_threshold=args.sigma_threshold,
        min_area=args.min_area,
        max_stars=args.max_stars,
        pattern_checking_stars=args.pattern_checking_stars,
        match_radius=args.match_radius,
        match_threshold=args.match_threshold,
        solve_timeout_ms=args.solve_timeout_ms,
        attitude_method=args.attitude_method,
    )
    try:
        result = StarTrackerPipeline(config).run(args.image)
    except Exception as exc:
        result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}

    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
