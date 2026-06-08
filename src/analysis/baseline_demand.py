"""
Baseline demand analysis for Sydney warehouses without solar generation.

Establishes typical consumption profiles using:
  - NEM NSW region 30-minute interval data
  - NABERS Energy warehouse benchmarks (per m² intensity)
  - Statistical methods for outlier-robust baseline estimation
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# NABERS / DCCEE commercial building energy intensity benchmarks
# Source: DCCEE Commercial Building Disclosure – warehouse sector
NABERS_BENCHMARK_KWH_PER_SQM_YEAR = {
    "cold_storage": 450,
    "refrigerated_distribution": 320,
    "ambient_logistics": 85,
    "light_industrial": 120,
    "general_warehouse": 75,
}

DEFAULT_WAREHOUSE_TYPE = "general_warehouse"


class BaselineDemandAnalyser:
    """
    Compute energy baseline from NEM interval data and warehouse floor area.

    Two complementary approaches:
      1. Statistical rolling baseline from measured NEM data.
      2. Benchmark-derived estimate when measured data is unavailable.
    """

    def __init__(
        self,
        nem_data: Optional[pd.DataFrame] = None,
        warehouse_area_sqm: float = 1_000.0,
        warehouse_type: str = DEFAULT_WAREHOUSE_TYPE,
    ) -> None:
        self.nem_data = nem_data
        self.warehouse_area_sqm = warehouse_area_sqm
        self.warehouse_type = warehouse_type

    # ------------------------------------------------------------------
    # Measured baseline (NEM data)
    # ------------------------------------------------------------------

    def rolling_baseline(
        self,
        window_days: int = 14,
        percentile: float = 0.10,
    ) -> pd.Series:
        """
        Rolling *percentile* baseline over *window_days* of NEM interval data.

        Uses a low percentile (default 10th) to capture the true baseline
        consumption floor, excluding operating peaks.
        """
        if self.nem_data is None or self.nem_data.empty:
            raise ValueError("NEM data is required for rolling_baseline()")

        demand_col = self._demand_column()
        series = self.nem_data[demand_col].dropna()
        rolling_window = window_days * 48  # 30-min intervals per day
        baseline = series.rolling(window=rolling_window, min_periods=24).quantile(percentile)
        baseline.name = "baseline_mw"
        return baseline

    def typical_daily_profile(self) -> pd.DataFrame:
        """
        Return average demand for each 30-minute slot across all weekdays/weekends.
        Useful for spotting operational patterns (shift start/end, HVAC cycling).
        """
        if self.nem_data is None or self.nem_data.empty:
            raise ValueError("NEM data is required for typical_daily_profile()")

        demand_col = self._demand_column()
        df = self.nem_data[[demand_col]].copy()
        df["time_of_day"] = df.index.time
        df["day_type"] = np.where(df.index.dayofweek < 5, "weekday", "weekend")
        profile = (
            df.groupby(["day_type", "time_of_day"])[demand_col]
            .agg(["mean", "median", "std"])
            .rename(columns={"mean": "mean_mw", "median": "median_mw", "std": "std_mw"})
        )
        return profile

    def detect_anomalies(self, z_threshold: float = 3.5) -> pd.DataFrame:
        """Flag intervals where demand deviates more than *z_threshold* standard deviations."""
        if self.nem_data is None or self.nem_data.empty:
            raise ValueError("NEM data is required for detect_anomalies()")

        demand_col = self._demand_column()
        series = self.nem_data[demand_col].dropna()
        z_scores = np.abs(stats.zscore(series))
        anomalies = self.nem_data[demand_col][z_scores > z_threshold].to_frame()
        anomalies["z_score"] = z_scores[z_scores > z_threshold]
        return anomalies

    # ------------------------------------------------------------------
    # Benchmark-derived estimate
    # ------------------------------------------------------------------

    def benchmark_annual_kwh(self) -> float:
        """Estimate annual consumption using NABERS intensity benchmarks."""
        intensity = NABERS_BENCHMARK_KWH_PER_SQM_YEAR.get(
            self.warehouse_type,
            NABERS_BENCHMARK_KWH_PER_SQM_YEAR[DEFAULT_WAREHOUSE_TYPE],
        )
        return intensity * self.warehouse_area_sqm

    def benchmark_interval_kw(self) -> pd.DataFrame:
        """
        Generate a synthetic 30-minute profile for one year derived from NABERS benchmark.
        Load shape follows a simplified warehouse pattern:
          - 07:00–18:00 weekdays: peak load
          - 18:00–22:00 weekdays: medium load
          - Off hours / weekends: baseload (lighting, standby, refrigeration)
        """
        index = pd.date_range("2024-01-01", "2024-12-31 23:30", freq="30min", tz="Australia/Sydney")
        annual_kwh = self.benchmark_annual_kwh()
        # Distribute load using shape factors
        shape = pd.Series(1.0, index=index)
        is_weekday = index.dayofweek < 5
        hour = index.hour
        shape.loc[is_weekday & (hour >= 7) & (hour < 18)] = 2.4
        shape.loc[is_weekday & (hour >= 18) & (hour < 22)] = 1.5
        # Normalize so total equals benchmark annual kWh
        kwh_per_interval = (shape / shape.sum()) * annual_kwh
        kw_demand = kwh_per_interval / 0.5  # 30-min interval → kW
        return pd.DataFrame({"demand_kw": kw_demand})

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return key baseline metrics as a plain dict."""
        benchmark = self.benchmark_annual_kwh()
        result = {
            "warehouse_area_sqm": self.warehouse_area_sqm,
            "warehouse_type": self.warehouse_type,
            "benchmark_annual_kwh": round(benchmark, 0),
            "benchmark_intensity_kwh_per_sqm": NABERS_BENCHMARK_KWH_PER_SQM_YEAR.get(
                self.warehouse_type, NABERS_BENCHMARK_KWH_PER_SQM_YEAR[DEFAULT_WAREHOUSE_TYPE]
            ),
            "avg_daily_kwh": round(benchmark / 365, 1),
            "avg_demand_kw": round(benchmark / (365 * 24), 2),
        }
        if self.nem_data is not None and not self.nem_data.empty:
            demand_col = self._demand_column()
            series = self.nem_data[demand_col].dropna()
            result.update(
                {
                    "nem_mean_mw": round(series.mean(), 3),
                    "nem_p10_mw": round(series.quantile(0.10), 3),
                    "nem_p90_mw": round(series.quantile(0.90), 3),
                    "nem_max_mw": round(series.max(), 3),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _demand_column(self) -> str:
        for candidate in ("total_demand_mw", "demand_and_nonschedgen_mw"):
            if self.nem_data is not None and candidate in self.nem_data.columns:
                return candidate
        raise KeyError("No recognisable demand column in NEM data")
