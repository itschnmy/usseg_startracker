"""Systematic, label-free image03 solver diagnostics; not an accuracy benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from models.pipeline import PipelineConfig, StarTrackerPipeline


ROOT = Path(__file__).resolve().parents[1]


def read_records(path):
    records = []
    invalid = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            invalid += 1
    return records, invalid


def trial_key(record):
    return (record["profile"], record["selected_count"], record["mirrored"],
            record["fov_start_deg"], record.get("pixel_aspect_scale_x", 1), record["database"])


def summarize(path):
    records, invalid = read_records(path)
    summary = {"trials":len(records), "solved":sum(r["status"] == "ok" for r in records),
               "runtime_errors":sum(r["status"] == "runtime_error" for r in records),
               "sum_trial_seconds":sum(r["elapsed_ms"] for r in records)/1000,
               "invalid_lines":invalid}
    path.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=float, default=1000)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--aspect", action="store_true", help="Diagnostic virtual pixel-aspect hypotheses")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed trials; write a new shard, preserving interrupted logs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = "aspect" if args.aspect else "extended" if args.extended else "coarse"
    path = args.output_dir / f"{phase}.jsonl"
    if args.summarize_only:
        print(summarize(path)); return
    original_path = path
    prior_paths = [path] + sorted(args.output_dir.glob(f"{phase}-resume-*.jsonl"))
    if path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh directory")
    completed = set()
    if args.resume:
        for previous in prior_paths:
            if previous.exists():
                records, invalid = read_records(previous)
                completed.update(trial_key(r) for r in records)
                print(f"Resume: {previous.name}: {len(records)} records, {invalid} invalid lines (preserved)",flush=True)
        index = 1
        while path.exists():
            path = args.output_dir / f"{phase}-resume-{index:02d}.jsonl"
            index += 1
    cv2.setNumThreads(1)
    source = ROOT/"dataset/03.bmp"
    color = cv2.imread(str(source))
    if color is None:
        raise ValueError(f"Cannot read {source}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    base = PipelineConfig.from_json(ROOT/"configs/examples/image03.json")
    engines = {
        "10-30": StarTrackerPipeline(base),
        "45": StarTrackerPipeline(replace(base, database=ROOT/"data/database-fov45.npz", fov_deg=45)),
    }
    profiles = []
    for kernel, sigma in ((0,3), (0,6), (15,3), (31,3), (31,5), (31,8), (51,3)):
        profiles.append((f"gray-k{kernel}-sigma{sigma}", gray,
                         replace(base, use_tophat=kernel>0, tophat_kernel_size=kernel or 7,
                                 sigma_threshold=sigma, border_margin_px=24)))
    profiles.extend([
        ("median3-k31", cv2.medianBlur(gray,3), replace(base, min_area=2, border_margin_px=24)),
        ("green-k31", color[:,:,1], replace(base, border_margin_px=24)),
    ])
    if args.extended:
        profiles = [p for p in profiles if p[0] in ("gray-k0-sigma6", "gray-k31-sigma5", "median3-k31")]
    if args.aspect:
        profiles = [p for p in profiles if p[0] in ("gray-k0-sigma6", "gray-k31-sigma8")]
    meta = {
        "image": str(source), "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "label_usage": "none: no label file, sky center or reference used",
        "opencv_threads": cv2.getNumThreads(), "timeout_ms": args.timeout_ms,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT/"models").rglob("*.py"))},
        "database_sha256": {name: hashlib.sha256(p.config.database.read_bytes()).hexdigest()
                            for name,p in engines.items()},
    }
    if args.resume and original_path.with_suffix(".run.json").exists():
        old_meta = json.loads(original_path.with_suffix(".run.json").read_text())
        for key in ("image_sha256", "source_sha256", "database_sha256", "timeout_ms"):
            if old_meta[key] != meta[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        meta["resumed_from"] = [str(p) for p in prior_paths if p.exists()]
    path.with_suffix(".run.json").write_text(json.dumps(meta,indent=2)+"\n", encoding="utf-8")
    count = solved = 0
    start = perf_counter()
    with path.open("x", encoding="utf-8") as output:
        for name, image, config in profiles:
            engines["10-30"].config = config
            stars = engines["10-30"].detect_stars(image, limit=False)
            all_xy = np.asarray([s.position for s in stars]).reshape(-1,2)
            counts = (12,20) if args.extended else (8,12,20)
            if args.extended or args.aspect:
                counts = tuple(dict.fromkeys(min(n,len(stars)) for n in counts))
            fovs = (10.5,17.435,28,45) if args.extended else (6,10.5,17.435,24,30,38,45,60)
            if args.aspect:
                fovs = (10.5,17.435,28,45)
            scales = (.5,.75,4/3,2) if args.aspect else (1,)
            for n, mirrored, fov, scale in itertools.product(counts, (False,True), fovs, scales):
                xy = all_xy[:n].copy()
                if mirrored:
                    xy[:,0] = image.shape[1]-1-xy[:,0]
                xy[:,0] *= scale
                size = (image.shape[0],round(image.shape[1]*scale))
                database = "45" if fov >= 38 else "10-30"
                if (name, len(xy), mirrored, fov, scale, database) in completed:
                    continue
                engine = engines[database].solver
                result = {"RA": None}
                error = None
                t0 = perf_counter()
                if len(xy) >= 4:
                    try:
                        result = engine.solve_from_centroids(
                            xy, size, fov_estimate=fov, fov_max_error=12,
                            pattern_checking_stars=min(n,len(xy)),
                            match_radius=config.match_radius, match_threshold=config.match_threshold,
                            solve_timeout=args.timeout_ms, return_matches=True, return_visual=False)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                record = {
                    "profile": name, "detector_config": {**asdict(config),"database":str(config.database)},
                    "database": database, "mirrored": mirrored, "fov_start_deg": fov,
                    "pixel_aspect_scale_x":scale,"solver_image_size_hw":size,
                    "selected_count": len(xy), "available_components":len(stars),
                    "centroids_xy": xy.tolist(), "elapsed_ms":(perf_counter()-t0)*1000,
                    "status": "runtime_error" if error else "ok" if result.get("RA") is not None else "no_solve",
                    "error":error, "solver_result":result,
                }
                output.write(json.dumps(record,allow_nan=False)+"\n"); output.flush()
                count += 1; solved += record["status"] == "ok"
                if count%8 == 0 or record["status"] == "ok":
                    print(f"{count} trials; solved={solved}; profile={name}; elapsed={perf_counter()-start:.1f}s",flush=True)
        print(summarize(path),flush=True)


if __name__ == "__main__":
    main()
