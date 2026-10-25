from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np

from models.identifier.plate_solver import PlateSolver, _compute_vectors
from models.detector.star_detector import StarDetector
from models.attitude import AttitudeDeterminator


def _radec_to_unit_vectors(stars: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(stars[:, 0])
    dec = np.deg2rad(stars[:, 1])
    return np.column_stack(
        (np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec))
    )


@dataclass(frozen=True)
class PipelineConfig:
    database: Path
    fov_deg: float
    fov_max_error_deg: float = 5.0
    sigma_threshold: float = 3.0
    min_area: int = 1
    max_area: int = 50
    use_tophat: bool = True
    tophat_kernel_size: int = 7
    component_mask: bool = False
    border_margin_px: int = 0
    star_ranking: str = "peak"
    max_stars: int = 20
    pattern_checking_stars: int = 12
    max_input_stars: int | None = None
    verification_catalog_stars: int | None = None
    match_radius: float = 0.015
    match_threshold: float = 1e-3
    solve_timeout_ms: float | None = 600.0
    attitude_method: str = "SVD"

    def __post_init__(self):
        for name in ("max_input_stars", "verification_catalog_stars"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 4):
                raise ValueError(f"{name} must be an integer >= 4, or null")
        if not math.isfinite(self.fov_deg) or not 0 < self.fov_deg < 180:
            raise ValueError("fov_deg must be horizontal FOV between 0 and 180 degrees")
        if not math.isfinite(self.fov_max_error_deg) or self.fov_max_error_deg < 0:
            raise ValueError("fov_max_error_deg must be finite and nonnegative")
        if not math.isfinite(self.sigma_threshold) or self.sigma_threshold <= 0:
            raise ValueError("sigma_threshold must be finite and positive")
        for name in ("min_area", "max_area", "max_stars", "pattern_checking_stars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_area < self.min_area:
            raise ValueError("max_area must not be smaller than min_area")
        if self.max_stars < 4 or self.pattern_checking_stars < 4:
            raise ValueError("max_stars and pattern_checking_stars must be at least 4")
        if not isinstance(self.border_margin_px, int) or self.border_margin_px < 0:
            raise ValueError("border_margin_px must be a nonnegative integer")
        if (not isinstance(self.tophat_kernel_size, int) or self.tophat_kernel_size < 1
                or self.tophat_kernel_size % 2 == 0):
            raise ValueError("tophat_kernel_size must be a positive odd integer")
        for name in ("match_radius", "match_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.solve_timeout_ms is not None and (
                not math.isfinite(self.solve_timeout_ms) or self.solve_timeout_ms <= 0):
            raise ValueError("solve_timeout_ms must be positive and finite, or null")
        if self.attitude_method.upper() not in ("SVD", "QUEST", "DAVENPORT", "TRIAD"):
            raise ValueError("unsupported attitude_method")
        if self.star_ranking not in ("peak", "flux"):
            raise ValueError("star_ranking must be peak or flux")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        import json

        path = Path(path).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("config must be a JSON object")
        if "database" in values:
            values["database"] = (path.parent / values["database"]).resolve()
        return cls(**values)


class StarTrackerPipeline:
    """Image -> centroid -> Tetra plate solve -> Wahba attitude pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        db_path = Path(config.database).resolve()
        if not db_path.exists():
            candidate = db_path.parent.parent / "data" / db_path.name
            if candidate.exists():
                db_path = candidate
        self.solver = PlateSolver(db_path)
        props = self.solver.database_properties
        min_fov = float(props["min_fov"])
        max_fov = float(props["max_fov"])
        lower = config.fov_deg - config.fov_max_error_deg
        upper = config.fov_deg + config.fov_max_error_deg
        if upper < min_fov or lower > max_fov:
            raise ValueError(
                f"FOV search range [{lower}, {upper}] deg does not overlap "
                f"database range [{min_fov}, {max_fov}] deg"
            )

    def run(self, image_path: str | Path) -> dict[str, Any]:
        total_start = perf_counter_ns()

        load_start = perf_counter_ns()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        load_ns = perf_counter_ns() - load_start
        if image is None:
            return self._failure("error", f"could not read image: {image_path}", total_start, load_ns)

        return self._run_image(
            image,
            source=str(Path(image_path).resolve()),
            total_start=total_start,
            load_ns=load_ns,
        )

    def run_array(self, image: np.ndarray, *, source: str = "<array>") -> dict[str, Any]:
        """Run the unchanged pipeline on an already-decoded image array.

        This is an I/O adapter for scientific containers such as HDF5.  Detection,
        plate solving, and attitude estimation use exactly the same code path as
        :meth:`run`; decoding time is intentionally reported as zero so an
        evaluator can measure container loading separately.
        """
        total_start = perf_counter_ns()
        if not isinstance(image, np.ndarray) or image.size == 0:
            return self._failure("error", "image array is empty", total_start)
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
            return self._failure("error", "expected grayscale or BGR image array", total_start)
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            return self._failure("error", "image array must contain finite numeric pixels", total_start)
        return self._run_image(image, source=source, total_start=total_start, load_ns=0)

    def detect_stars(self, image: np.ndarray, *, limit: bool = True):
        """Shared configured preprocessing/detection for inference and diagnostics."""
        height, width = image.shape[:2]
        focal_px = (width / 2.0) / np.tan(np.deg2rad(self.config.fov_deg) / 2.0)
        detector = StarDetector(
            sigma_threshold=self.config.sigma_threshold,
            min_area=self.config.min_area,
            max_area=self.config.max_area,
            use_tophat=self.config.use_tophat,
            tophat_kernel_size=self.config.tophat_kernel_size,
            component_mask=self.config.component_mask,
            border_margin_px=self.config.border_margin_px,
        )
        detector.set_intrinsics(width / 2.0, height / 2.0, focal_px)
        detected = detector.process(image)
        if self.config.star_ranking == "flux":
            detected.sort(key=lambda star: (star.intensity, star.peak), reverse=True)
        else:
            detected.sort(key=lambda star: (star.peak, star.intensity), reverse=True)
        return detected[:self.config.max_stars] if limit else detected

    def _run_image(
        self,
        image: np.ndarray,
        *,
        source: str,
        total_start: int,
        load_ns: int,
    ) -> dict[str, Any]:

        height, width = image.shape[:2]
        detect_start = perf_counter_ns()
        detected = self.detect_stars(image)
        detection_ns = perf_counter_ns() - detect_start
        centroids_xy = np.asarray(
            [star.position for star in detected], dtype=np.float64
        ).reshape((-1, 2))
        # This repository's PlateSolver wrapper accepts public (x, y) input and
        # converts it to Tetra3's internal (y, x) order itself.

        if len(detected) < 4:
            return self._failure(
                "no_solve",
                "fewer than four centroids",
                total_start,
                load_ns,
                detection_ns,
                num_centroids=len(detected),
                centroids_xy=centroids_xy,
            )

        solve_start = perf_counter_ns()
        solution = self.solver.solve_from_centroids(
            centroids_xy,
            (height, width),
            fov_estimate=self.config.fov_deg,
            fov_max_error=self.config.fov_max_error_deg,
            pattern_checking_stars=min(self.config.pattern_checking_stars, len(detected)),
            max_input_stars=self.config.max_input_stars,
            verification_catalog_stars=self.config.verification_catalog_stars,
            match_radius=self.config.match_radius,
            match_threshold=self.config.match_threshold,
            solve_timeout=self.config.solve_timeout_ms,
            return_matches=True,
            return_visual=False,
        )
        star_id_ns = perf_counter_ns() - solve_start

        if solution.get("RA") is None:
            return self._failure(
                "no_solve",
                "plate solver found no unique solution",
                total_start,
                load_ns,
                detection_ns,
                star_id_ns,
                num_centroids=len(detected),
                centroids_xy=centroids_xy,
            )

        attitude_start = perf_counter_ns()
        matched_centroids_yx = np.asarray(solution["matched_centroids"], dtype=np.float64)
        matched_stars = np.asarray(solution["matched_stars"], dtype=np.float64)
        body_vectors = _compute_vectors(
            matched_centroids_yx,
            (height, width),
            np.deg2rad(float(solution["FOV"])),
        ).T
        inertial_vectors = _radec_to_unit_vectors(matched_stars).T
        quaternion_internal = AttitudeDeterminator(self.config.attitude_method).estimate(
            body_vectors, inertial_vectors
        )
        quaternion_internal = np.asarray(quaternion_internal, dtype=np.float64)
        quaternion_internal /= np.linalg.norm(quaternion_internal)
        if quaternion_internal[0] < 0:
            quaternion_internal = -quaternion_internal

        # SVD returns body-to-inertial, while QUEST/Davenport/TRIAD return
        # inertial-to-body. Normalize the public direction explicitly.
        quaternion_lost = quaternion_internal.copy()
        if self.config.attitude_method.upper() == "SVD" and body_vectors.shape[1] > 2:
            quaternion_lost[1:] *= -1
        if quaternion_lost[0] < 0:
            quaternion_lost = -quaternion_lost
        quaternion_body_to_inertial = quaternion_lost * np.array([1, -1, -1, -1])
        attitude_ns = perf_counter_ns() - attitude_start
        total_ns = perf_counter_ns() - total_start

        timings = {
            "image_load_ns": load_ns,
            "detection_ns": detection_ns,
            "star_id_ns": star_id_ns,
            "attitude_ns": attitude_ns,
            "pipeline_ns": detection_ns + star_id_ns + attitude_ns,
            "total_ns": total_ns,
        }
        return {
            "status": "ok",
            "message": None,
            "image": source,
            "image_size_hw": [height, width],
            "num_centroids": len(detected),
            "num_matches": int(solution["Matches"]),
            "centroids_xy": centroids_xy.tolist(),
            "matched_centroids_yx": solution["matched_centroids"],
            "matched_stars_radec": [s[:2] for s in solution["matched_stars"]],
            "matched_catalog_ids": solution.get("matched_catID"),
            "quaternion_wxyz": quaternion_lost.tolist(),
            "quaternion_lost_wxyz": quaternion_lost.tolist(),
            "quaternion_inertial_to_body_wxyz": quaternion_lost.tolist(),
            "quaternion_body_to_inertial_wxyz": quaternion_body_to_inertial.tolist(),
            "quaternion_internal_wxyz": quaternion_internal.tolist(),
            "ra_deg": float(solution["RA"]),
            "dec_deg": float(solution["Dec"]),
            "roll_deg": float(solution["Roll"]),
            "solved_fov_deg": float(solution["FOV"]),
            "rmse_arcsec": float(solution["RMSE"]),
            "mismatch_probability": float(solution["Prob"]),
            "fps": 1e9 / total_ns if total_ns else None,
            "timings": timings,
            "config": {**asdict(self.config), "database": str(self.config.database)},
        }

    def _failure(
        self,
        status: str,
        message: str,
        total_start: int,
        load_ns: int = 0,
        detection_ns: int = 0,
        star_id_ns: int = 0,
        *,
        num_centroids: int = 0,
        centroids_xy: np.ndarray | None = None,
    ) -> dict[str, Any]:
        total_ns = perf_counter_ns() - total_start
        return {
            "status": status,
            "message": message,
            "config": {**asdict(self.config), "database": str(self.config.database)},
            "num_centroids": num_centroids,
            "num_matches": 0,
            "centroids_xy": centroids_xy.tolist() if centroids_xy is not None else [],
            "quaternion_wxyz": None,
            "fps": 1e9 / total_ns if total_ns else None,
            "timings": {
                "image_load_ns": load_ns,
                "detection_ns": detection_ns,
                "star_id_ns": star_id_ns,
                "attitude_ns": 0,
                "pipeline_ns": detection_ns + star_id_ns,
                "total_ns": total_ns,
            },
        }
