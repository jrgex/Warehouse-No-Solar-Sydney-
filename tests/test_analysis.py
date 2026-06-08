"""
Unit tests for analysis and cost estimation modules.
Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.baseline_demand import BaselineDemandAnalyser, NABERS_BENCHMARK_KWH_PER_SQM_YEAR
from src.analysis.peak_offpeak import PeakOffPeakAnalyser
from src.cost_estimation.tariff_calculator import TariffCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_profile() -> pd.DataFrame:
    """One week of 30-minute synthetic load data, Sydney local time."""
    idx = pd.date_range(
        "2024-01-15",
        periods=7 * 48,
        freq="30min",
        tz="Australia/Sydney",
    )
    rng = np.random.default_rng(42)
    # Realistic warehouse load: daytime ~120 kW, night ~40 kW
    hour = idx.hour
    base = np.where((hour >= 7) & (hour < 18), 120.0, 40.0)
    noise = rng.normal(0, 5, len(idx))
    return pd.DataFrame({"demand_kw": np.maximum(base + noise, 10.0)}, index=idx)


@pytest.fixture
def nem_data() -> pd.DataFrame:
    """Minimal NEM-style DataFrame for baseline tests."""
    idx = pd.date_range("2024-01-01", periods=48 * 30, freq="30min")
    rng = np.random.default_rng(0)
    demand = 8000 + rng.normal(0, 400, len(idx))
    return pd.DataFrame({"total_demand_mw": demand}, index=idx)


# ---------------------------------------------------------------------------
# BaselineDemandAnalyser
# ---------------------------------------------------------------------------

class TestBaselineDemandAnalyser:
    def test_benchmark_kwh_scales_with_area(self):
        a1 = BaselineDemandAnalyser(warehouse_area_sqm=1_000)
        a2 = BaselineDemandAnalyser(warehouse_area_sqm=2_000)
        assert a2.benchmark_annual_kwh() == pytest.approx(2 * a1.benchmark_annual_kwh())

    def test_benchmark_kwh_uses_correct_intensity(self):
        area = 1_500
        wtype = "cold_storage"
        expected = NABERS_BENCHMARK_KWH_PER_SQM_YEAR[wtype] * area
        analyser = BaselineDemandAnalyser(warehouse_area_sqm=area, warehouse_type=wtype)
        assert analyser.benchmark_annual_kwh() == pytest.approx(expected)

    def test_benchmark_interval_length(self):
        analyser = BaselineDemandAnalyser(warehouse_area_sqm=1_000)
        profile = analyser.benchmark_interval_kw()
        # 2024 is a leap year: 366 days * 48 intervals = 17568
        assert len(profile) == 17568

    def test_benchmark_total_matches_annual(self):
        analyser = BaselineDemandAnalyser(warehouse_area_sqm=2_000, warehouse_type="ambient_logistics")
        profile = analyser.benchmark_interval_kw()
        # kWh = kW * 0.5 h per interval
        total_kwh = (profile["demand_kw"] * 0.5).sum()
        expected = analyser.benchmark_annual_kwh()
        assert total_kwh == pytest.approx(expected, rel=0.01)

    def test_summary_keys(self, nem_data):
        analyser = BaselineDemandAnalyser(nem_data=nem_data, warehouse_area_sqm=1_000)
        summary = analyser.summary()
        for key in ("benchmark_annual_kwh", "nem_mean_mw", "nem_p10_mw"):
            assert key in summary

    def test_rolling_baseline_requires_nem_data(self):
        analyser = BaselineDemandAnalyser()
        with pytest.raises(ValueError):
            analyser.rolling_baseline()


# ---------------------------------------------------------------------------
# PeakOffPeakAnalyser
# ---------------------------------------------------------------------------

class TestPeakOffPeakAnalyser:
    def test_all_periods_present(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        assert set(pa.profile["tou_period"].unique()) <= {"peak", "shoulder", "offpeak"}

    def test_total_kwh_consistent(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        breakdown = pa.breakdown_by_period()
        manual_kwh = (small_profile["demand_kw"] * 0.5).sum()
        assert breakdown["total_kwh"].sum() == pytest.approx(manual_kwh, rel=0.001)

    def test_pct_sums_to_100(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        breakdown = pa.breakdown_by_period()
        assert breakdown["pct_of_total"].sum() == pytest.approx(100.0, abs=0.1)

    def test_peak_demand_events_length(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        events = pa.peak_demand_events(top_n=5)
        assert len(events) == 5

    def test_requires_datetimeindex(self):
        bad_df = pd.DataFrame({"demand_kw": [100, 200]}, index=[0, 1])
        with pytest.raises(TypeError):
            PeakOffPeakAnalyser(bad_df)

    def test_january_weekday_peak_period(self):
        # 2024-01-15 is a Monday (summer weekday) at 10:00 → peak
        idx = pd.date_range("2024-01-15 10:00", periods=1, freq="30min", tz="Australia/Sydney")
        df = pd.DataFrame({"demand_kw": [100.0]}, index=idx)
        pa = PeakOffPeakAnalyser(df)
        assert pa.profile["tou_period"].iloc[0] == "peak"

    def test_june_weekday_shoulder_period(self):
        # 2024-06-17 is a Monday (winter weekday) at 14:00 → shoulder
        idx = pd.date_range("2024-06-17 14:00", periods=1, freq="30min", tz="Australia/Sydney")
        df = pd.DataFrame({"demand_kw": [100.0]}, index=idx)
        pa = PeakOffPeakAnalyser(df)
        assert pa.profile["tou_period"].iloc[0] == "shoulder"

    def test_midnight_offpeak(self):
        # Any day at 01:00 → off-peak
        idx = pd.date_range("2024-03-15 01:00", periods=1, freq="30min", tz="Australia/Sydney")
        df = pd.DataFrame({"demand_kw": [50.0]}, index=idx)
        pa = PeakOffPeakAnalyser(df)
        assert pa.profile["tou_period"].iloc[0] == "offpeak"


# ---------------------------------------------------------------------------
# TariffCalculator
# ---------------------------------------------------------------------------

class TestTariffCalculator:
    def test_total_cost_positive(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa, roof_area_sqm=2_000)
        report = calc.annual_cost_report()
        assert report.total_annual_cost > 0

    def test_gst_is_10pct_of_subtotal(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa)
        report = calc.annual_cost_report()
        assert report.gst == pytest.approx(report.subtotal_excl_gst * 0.10, rel=0.001)

    def test_total_equals_subtotal_plus_gst(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa, roof_area_sqm=1_500)
        report = calc.annual_cost_report()
        assert report.total_annual_cost == pytest.approx(
            report.subtotal_excl_gst + report.gst, rel=0.001
        )

    def test_solar_fields_none_without_roof_area(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa, roof_area_sqm=0)
        report = calc.annual_cost_report()
        assert report.installable_kwp is None
        assert report.solar_annual_saving is None

    def test_solar_saving_less_than_total_bill(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa, roof_area_sqm=3_000)
        report = calc.annual_cost_report()
        assert report.solar_annual_saving < report.total_annual_cost

    def test_payback_calculated(self, small_profile):
        pa = PeakOffPeakAnalyser(small_profile)
        calc = TariffCalculator(pa, roof_area_sqm=2_000)
        report = calc.annual_cost_report()
        assert report.solar_simple_payback_years is not None
        assert report.solar_simple_payback_years > 0
