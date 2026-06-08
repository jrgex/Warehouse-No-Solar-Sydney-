"""
End-to-end analysis script — runs sample data through the full pipeline.

Usage:
    python scripts/run_analysis.py

For live detection (requires GOOGLE_MAPS_API_KEY in .env):
    python scripts/run_analysis.py --detect
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DATA_DIR
from src.analysis.baseline_demand import BaselineDemandAnalyser
from src.analysis.peak_offpeak import PeakOffPeakAnalyser
from src.cost_estimation.tariff_calculator import TariffCalculator
from src.data_ingestion.nem_loader import NEMLoader


def run_sample_pipeline(warehouse_csv: Path, nem_csv: Path) -> None:
    print("\n" + "=" * 60)
    print("  WAREHOUSE NO-SOLAR SYDNEY – SAMPLE ANALYSIS")
    print("=" * 60)

    # 1. Load warehouse registry
    warehouses = pd.read_csv(warehouse_csv)
    no_solar = warehouses[~warehouses["has_solar"]]
    print(f"\n[1] Warehouse registry loaded: {len(warehouses)} sites")
    print(f"    No-solar sites (analysis candidates): {len(no_solar)}")

    # 2. Load NEM data
    loader = NEMLoader()
    nem_df = loader.load_csv(nem_csv)
    print(f"\n[2] NEM data loaded: {len(nem_df)} intervals  ({nem_df.index.min()} – {nem_df.index.max()})")

    # 3. Analyse each no-solar warehouse individually
    print("\n[3] Per-warehouse cost estimates (benchmark load profiles)\n")
    print(f"  {'Name':<35} {'Area':>6} {'Ann. kWh':>10} {'Ann. Cost':>12} {'Solar Save':>12} {'Payback':>9}")
    print("  " + "-" * 90)

    for _, wh in no_solar.iterrows():
        ba = BaselineDemandAnalyser(
            warehouse_area_sqm=wh["estimated_area_sqm"],
            warehouse_type=wh.get("warehouse_type", "general_warehouse"),
        )
        profile = ba.benchmark_interval_kw()
        pa = PeakOffPeakAnalyser(profile)
        calc = TariffCalculator(pa, roof_area_sqm=wh["estimated_area_sqm"])
        r = calc.annual_cost_report()
        pb_str = f"{r.solar_simple_payback_years:.1f} yr" if r.solar_simple_payback_years else "  N/A"
        save_str = f"${r.solar_annual_saving:>10,.0f}" if r.solar_annual_saving else "       N/A"
        print(
            f"  {wh['name']:<35} {wh['estimated_area_sqm']:>6,.0f}"
            f" {r.total_kwh:>10,.0f} ${r.total_annual_cost:>11,.0f} {save_str} {pb_str:>9}"
        )

    # 4. Detailed report for the largest warehouse
    largest = no_solar.sort_values("estimated_area_sqm", ascending=False).iloc[0]
    print(f"\n[4] Detailed report: {largest['name']} ({largest['estimated_area_sqm']:,.0f} m²)")
    ba = BaselineDemandAnalyser(
        warehouse_area_sqm=largest["estimated_area_sqm"],
        warehouse_type=largest.get("warehouse_type", "general_warehouse"),
    )
    profile = ba.benchmark_interval_kw()
    pa = PeakOffPeakAnalyser(profile)
    calc = TariffCalculator(pa, roof_area_sqm=largest["estimated_area_sqm"])
    calc.print_report()

    # 5. NSW grid context from NEM data
    print("[5] NSW Grid Summary (sample day from NEM data)")
    print(f"    Mean demand: {nem_df['total_demand_mw'].mean():.1f} MW")
    print(f"    Peak demand: {nem_df['total_demand_mw'].max():.1f} MW")
    print(f"    Min  demand: {nem_df['total_demand_mw'].min():.1f} MW\n")


def run_live_detection() -> None:
    from config.settings import GOOGLE_MAPS_API_KEY
    from src.data_ingestion.warehouse_detector import WarehouseDetector

    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: Set GOOGLE_MAPS_API_KEY in your .env file to run live detection.")
        sys.exit(1)

    print("\nStarting live warehouse detection across Greater Sydney…")
    detector = WarehouseDetector()
    results = detector.scan_sydney(max_results=50)
    print(f"\nFound {len(results)} no-solar warehouses (>500 m²)")
    if not results.empty:
        print(results[["name", "address", "estimated_area_sqm", "solar_coverage_pct"]].to_string(index=False))
        out = DATA_DIR / "detected_warehouses.csv"
        results.to_csv(out, index=False)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse No-Solar Sydney analysis")
    parser.add_argument("--detect", action="store_true", help="Run live Google Maps detection")
    args = parser.parse_args()

    if args.detect:
        run_live_detection()
    else:
        run_sample_pipeline(
            warehouse_csv=DATA_DIR / "sample_warehouses.csv",
            nem_csv=DATA_DIR / "sample_nem_nsw_2024.csv",
        )
