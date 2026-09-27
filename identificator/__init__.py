"""Backward-compatibility shim for legacy identificator package."""
from models.identifier.plate_solver import PlateSolver, _compute_vectors
from models.detector.star_detector import StarDetector

__all__ = ["PlateSolver", "_compute_vectors", "StarDetector"]
