"""Backward-compatibility shim for legacy src.AttitudeDeterminator."""
from models.attitude import (
    AttitudeDeterminator,
    QUESTEstimator,
    DavenportQEstimator,
    TRIADEstimator,
    SVDEstimator,
    MEKFEstimator,
    AttitudeControlSystem,
    ADCSMode,
    RelativeAttitudeDeterminator,
)
from models.attitude import estimators, attitude_determinator, mekf
from utils import math_utils

__all__ = [
    "AttitudeDeterminator",
    "QUESTEstimator",
    "DavenportQEstimator",
    "TRIADEstimator",
    "SVDEstimator",
    "MEKFEstimator",
    "AttitudeControlSystem",
    "ADCSMode",
    "RelativeAttitudeDeterminator",
    "estimators",
    "attitude_determinator",
    "mekf",
    "math_utils",
]
