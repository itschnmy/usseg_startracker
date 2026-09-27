"""Default configuration parameters for USSEG Star Tracker."""
from dataclasses import dataclass

@dataclass
class DefaultConfig:
    fov_deg: float = 26.0
    fov_max_error_deg: float = 5.0
    sigma_threshold: float = 3.0
    min_area: int = 1
    max_area: int = 50
    use_tophat: bool = True
    tophat_kernel_size: int = 7
    max_stars: int = 20
    pattern_checking_stars: int = 12
    solve_timeout_ms: float = 600.0
    attitude_method: str = "SVD"
