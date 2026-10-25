from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from .pipeline import PipelineConfig, StarTrackerPipeline


def build_parser(config: PipelineConfig | None = None) -> argparse.ArgumentParser:
    defaults = {field.name: field.default for field in fields(PipelineConfig)}
    if config is not None:
        defaults.update(vars(config))
    parser = argparse.ArgumentParser(description="Run the unified USSEG star-tracker pipeline")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Pipeline JSON; database path is relative to this file")
    parser.add_argument("--database", required=config is None, type=Path,
                        default=config.database if config else None)
    parser.add_argument("--fov", required=config is None, type=float,
                        default=config.fov_deg if config else None,
                        help="Horizontal FOV in degrees (not diagonal or vertical)")
    parser.add_argument("--fov-max-error", type=float, default=defaults["fov_max_error_deg"])
    parser.add_argument("--sigma-threshold", type=float, default=defaults["sigma_threshold"])
    parser.add_argument("--min-area", type=int, default=defaults["min_area"])
    parser.add_argument("--max-area", type=int, default=defaults["max_area"])
    parser.add_argument("--tophat-kernel-size", type=int, default=defaults["tophat_kernel_size"])
    parser.add_argument("--use-tophat", action=argparse.BooleanOptionalAction, default=defaults["use_tophat"])
    parser.add_argument("--component-mask", action=argparse.BooleanOptionalAction, default=defaults["component_mask"])
    parser.add_argument("--border-margin-px", type=int, default=defaults["border_margin_px"])
    parser.add_argument("--star-ranking", choices=("peak", "flux"), default=defaults["star_ranking"])
    parser.add_argument("--max-stars", type=int, default=defaults["max_stars"])
    parser.add_argument("--max-input-stars", type=int, default=defaults["max_input_stars"])
    parser.add_argument("--verification-catalog-stars", type=int, default=defaults["verification_catalog_stars"])
    parser.add_argument("--pattern-checking-stars", type=int, default=defaults["pattern_checking_stars"])
    parser.add_argument("--match-radius", type=float, default=defaults["match_radius"])
    parser.add_argument("--match-threshold", type=float, default=defaults["match_threshold"])
    parser.add_argument("--solve-timeout-ms", type=float, default=defaults["solve_timeout_ms"])
    parser.add_argument(
        "--attitude-method",
        choices=("SVD", "QUEST", "DAVENPORT", "TRIAD"),
        default=defaults["attitude_method"],
    )
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    try:
        preset = PipelineConfig.from_json(known.config) if known.config else None
    except (OSError, ValueError, TypeError) as exc:
        preliminary.error(f"invalid config: {exc}")
    args = build_parser(preset).parse_args(argv)
    try:
        # Keep every config field forwarded, including future additions. Only
        # the two historical CLI names differ from the dataclass names.
        values = vars(args).copy()
        values["fov_deg"] = args.fov
        values["fov_max_error_deg"] = args.fov_max_error
        config = PipelineConfig(**{field.name: values[field.name] for field in fields(PipelineConfig)})
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
