"""Configuration dataclasses for GBQR model."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters.

    Attributes:
        model_name: Name of the model, used for output file naming.
        use_ilinet: Whether to include ILINet data as supplementary training.
        use_flusurvnet: Whether to include FluSurvNet data as supplementary training.
        use_nhsn: Whether to include NHSN data as supplementary training.
        use_nssp_extra: Whether to include additional NSSP locations (beyond MCHub).
        num_bags: Number of bootstrap bags for bagging.
        bag_frac_samples: Fraction of seasons to sample per bag.
        incl_level_feats: Whether to include level features in the model.
        power_transform: Power transform to apply ("4rt" or None).
        fit_locations_separately: Whether to fit separate models per location.
        drop_seasons: List of seasons to exclude from training.
    """

    model_name: str = "gbqr"

    # Supplementary training data sources (individual toggles)
    # Note: MCHub data is always loaded as the primary source
    use_ilinet: bool = False
    use_flusurvnet: bool = False
    use_nhsn: bool = False
    use_nssp_extra: bool = False

    # Weather data as additional features (not incidence data)
    use_weather: bool = False
    weather_file: Optional[Path] = None  # Path to weather CSV, or None for default
    weather_features: List[str] = field(
        default_factory=lambda: ["temp_avg_f", "humidity_abs_gm3"]
    )
    weather_lags: List[int] = field(
        default_factory=lambda: [1, 2]  # 1-week and 2-week lags
    )
    weather_rolling_windows: List[int] = field(
        default_factory=lambda: [3]  # 3-week trailing average
    )

    # Bagging parameters
    num_bags: int = 100
    bag_frac_samples: float = 0.7

    # Feature configuration
    incl_level_feats: bool = True
    power_transform: Optional[str] = "4rt"

    # Model behavior
    fit_locations_separately: bool = False

    # Season filtering
    drop_seasons: List[str] = field(
        default_factory=lambda: ["2020/21", "2021/22"]
    )


@dataclass
class RunConfig:
    """Runtime configuration for a forecast run.

    Attributes:
        ref_date: Reference date for the forecast (determines data version).
        hub_root: Path to the MCHub repository root.
        output_root: Path where predictions will be saved.
        max_horizon: Maximum forecast horizon in weeks.
        q_levels: Quantile levels for prediction.
        q_labels: String labels for quantile levels (for output).
    """

    ref_date: datetime.date
    hub_root: Path
    output_root: Path

    max_horizon: int = 4
    q_levels: List[float] = field(
        default_factory=lambda: [
            0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975
        ]
    )
    q_labels: List[str] = field(
        default_factory=lambda: [
            "0.025", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.975"
        ]
    )
