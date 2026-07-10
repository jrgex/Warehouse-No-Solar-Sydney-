"""
Electricity cost estimation for Sydney warehouses using Ausgrid network tariffs.

Tariff structure: Ausgrid EA305 – General Large Business (2024-25)
Includes:
  - Network charges (usage + demand + daily supply)
  - Retail energy margin
  - Environmental levies (LRET, SRES, ESC)
  - GST

Solar savings calculation uses NREL PVWatts assumptions for Sydney (−33.9°).
"""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import (
    AUSGRID_TARIFFS,
    PANEL_KWP_PER_SQM_ROOF,
    RETAIL_ENERGY_MARGIN_KWH,
    SOLAR_COST_PER_KWP,
    SOLAR_KWH_PER_KWP_PER_DAY,
)
from src.analysis.peak_offpeak import PeakOffPeakAnalyser

logger = logging.getLogger(__name__)

# Australian environmental levies (approximate, 2024 regulated values)
LARGE_SCALE_RENEWABLE_TARGET_KWH = 0.0128   # LRET certificate cost pass-through
SMALL_SCALE_RENEWABLE_KWH = 0.0065          # SRES (STCs)
NSW_ESC_KWH = 0.0048                        # NSW Energy Savings Scheme contribution
GST_RATE = 0.10


@dataclass
class CostBreakdown:
    """Full electricity cost breakdown for one warehouse over one year."""
    # Consumption
    peak_kwh: float
    shoulder_kwh: float
    offpeak_kwh: float
    total_kwh: float
    # Demand
    monthly_peak_kw: float
    # Charges (excl. GST)
    network_usage_cost: float
    network_demand_cost: float
    network_supply_cost: float
    retail_energy_cost: float
    environmental_levies: float
    # Totals
    subtotal_excl_gst: float
    gst: float
    total_annual_cost: float
    cost_per_kwh_effective: float
    # Solar opportunity
    installable_kwp: Optional[float] = None
    solar_annual_kwh: Optional[float] = None
    solar_annual_saving: Optional[float] = None
    solar_capex: Optional[float] = None
    solar_simple_payback_years: Optional[float] = None


class TariffCalculator:
    """
    Compute full electricity cost from a classified load profile.

    Usage
    -----
    >>> analyser = PeakOffPeakAnalyser(load_profile_df)
    >>> calc = TariffCalculator(analyser, roof_area_sqm=2500)
    >>> report = calc.annual_cost_report()
    >>> print(report)
    """

    def __init__(
        self,
        peak_offpeak_analyser: PeakOffPeakAnalyser,
        roof_area_sqm: float = 0.0,
        days_in_period: int = 365,
    ) -> None:
        self.analyser = peak_offpeak_analyser
        self.roof_area_sqm = roof_area_sqm
        self.days_in_period = days_in_period

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annual_cost_report(self) -> CostBreakdown:
        """Calculate and return a full CostBreakdown for the load profile."""
        breakdown = self.analyser.breakdown_by_period()
        peak_kwh = breakdown.loc["peak", "total_kwh"] if "peak" in breakdown.index else 0.0
        shoulder_kwh = breakdown.loc["shoulder", "total_kwh"] if "shoulder" in breakdown.index else 0.0
        offpeak_kwh = breakdown.loc["offpeak", "total_kwh"] if "offpeak" in breakdown.index else 0.0
        total_kwh = peak_kwh + shoulder_kwh + offpeak_kwh

        monthly_peak_kw = self.analyser.profile[self.analyser.demand_col].max()
        months = self.days_in_period / 30.44

        # Network usage charges
        network_usage = (
            peak_kwh * AUSGRID_TARIFFS["peak_rate_kwh"]
            + shoulder_kwh * AUSGRID_TARIFFS["shoulder_rate_kwh"]
            + offpeak_kwh * AUSGRID_TARIFFS["offpeak_rate_kwh"]
        )
        # Network demand charge (charged on monthly maximum demand)
        network_demand = monthly_peak_kw * AUSGRID_TARIFFS["demand_charge_kw"] * months
        # Daily supply charge
        network_supply = AUSGRID_TARIFFS["daily_supply_charge"] * self.days_in_period
        # Retail energy margin
        retail_energy = total_kwh * RETAIL_ENERGY_MARGIN_KWH
        # Environmental levies
        env_levies = total_kwh * (
            LARGE_SCALE_RENEWABLE_TARGET_KWH + SMALL_SCALE_RENEWABLE_KWH + NSW_ESC_KWH
        )

        subtotal = network_usage + network_demand + network_supply + retail_energy + env_levies
        gst = subtotal * GST_RATE
        total = subtotal + gst
        effective_rate = total / total_kwh if total_kwh else 0.0

        # Solar opportunity
        installable_kwp = solar_kwh = solar_saving = capex = payback = None
        if self.roof_area_sqm > 0:
            installable_kwp = self.roof_area_sqm * PANEL_KWP_PER_SQM_ROOF * 0.60  # 60% usable roof
            solar_kwh = installable_kwp * SOLAR_KWH_PER_KWP_PER_DAY * self.days_in_period
            solar_kwh = min(solar_kwh, total_kwh)  # can't generate more than consumed
            solar_saving = solar_kwh * effective_rate
            capex = installable_kwp * SOLAR_COST_PER_KWP
            payback = capex / solar_saving if solar_saving else None

        return CostBreakdown(
            peak_kwh=round(peak_kwh, 1),
            shoulder_kwh=round(shoulder_kwh, 1),
            offpeak_kwh=round(offpeak_kwh, 1),
            total_kwh=round(total_kwh, 1),
            monthly_peak_kw=round(monthly_peak_kw, 2),
            network_usage_cost=round(network_usage, 2),
            network_demand_cost=round(network_demand, 2),
            network_supply_cost=round(network_supply, 2),
            retail_energy_cost=round(retail_energy, 2),
            environmental_levies=round(env_levies, 2),
            subtotal_excl_gst=round(subtotal, 2),
            gst=round(gst, 2),
            total_annual_cost=round(total, 2),
            cost_per_kwh_effective=round(effective_rate, 4),
            installable_kwp=round(installable_kwp, 1) if installable_kwp else None,
            solar_annual_kwh=round(solar_kwh, 1) if solar_kwh else None,
            solar_annual_saving=round(solar_saving, 2) if solar_saving else None,
            solar_capex=round(capex, 0) if capex else None,
            solar_simple_payback_years=round(payback, 1) if payback else None,
        )

    def print_report(self) -> None:
        """Pretty-print the annual cost report to stdout."""
        r = self.annual_cost_report()
        sep = "─" * 52
        print(f"\n{'WAREHOUSE ELECTRICITY COST REPORT':^52}")
        print(f"{'Ausgrid EA305 | Sydney, NSW | 2024-25':^52}")
        print(sep)
        print(f"  Total consumption          {r.total_kwh:>12,.0f}  kWh")
        print(f"    Peak                     {r.peak_kwh:>12,.0f}  kWh")
        print(f"    Shoulder                 {r.shoulder_kwh:>12,.0f}  kWh")
        print(f"    Off-peak                 {r.offpeak_kwh:>12,.0f}  kWh")
        print(f"  Monthly peak demand        {r.monthly_peak_kw:>12,.1f}  kW")
        print(sep)
        print(f"  Network usage charges      ${r.network_usage_cost:>11,.2f}")
        print(f"  Network demand charges     ${r.network_demand_cost:>11,.2f}")
        print(f"  Network supply charges     ${r.network_supply_cost:>11,.2f}")
        print(f"  Retail energy margin       ${r.retail_energy_cost:>11,.2f}")
        print(f"  Environmental levies       ${r.environmental_levies:>11,.2f}")
        print(f"  Subtotal (excl. GST)       ${r.subtotal_excl_gst:>11,.2f}")
        print(f"  GST (10%)                  ${r.gst:>11,.2f}")
        print(sep)
        print(f"  TOTAL ANNUAL COST          ${r.total_annual_cost:>11,.2f}")
        print(f"  Effective rate             {r.cost_per_kwh_effective * 100:>11.2f}  c/kWh")
        if r.installable_kwp is not None:
            print(sep)
            print("  SOLAR OPPORTUNITY")
            print(f"  Installable capacity       {r.installable_kwp:>12,.1f}  kWp")
            print(f"  Estimated solar yield      {r.solar_annual_kwh:>12,.0f}  kWh/yr")
            print(f"  Annual bill saving         ${r.solar_annual_saving:>11,.2f}")
            print(f"  System capital cost        ${r.solar_capex:>11,.0f}")
            print(f"  Simple payback             {r.solar_simple_payback_years:>12.1f}  years")
        print(sep + "\n")

    # ------------------------------------------------------------------
    # Portfolio-level analysis
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_summary(
        warehouse_df: pd.DataFrame,
        baseline_analyser_class,
        tariff_calculator_class=None,
    ) -> pd.DataFrame:
        """
        Given a warehouse registry DataFrame (columns: name, estimated_area_sqm, warehouse_type),
        return per-warehouse cost estimates.
        """
        from src.analysis.baseline_demand import BaselineDemandAnalyser
        from src.cost_estimation.tariff_calculator import TariffCalculator

        if tariff_calculator_class is None:
            tariff_calculator_class = TariffCalculator

        rows = []
        for _, row in warehouse_df.iterrows():
            ba = BaselineDemandAnalyser(
                warehouse_area_sqm=row.get("estimated_area_sqm", 1000),
                warehouse_type=row.get("warehouse_type", "general_warehouse"),
            )
            profile = ba.benchmark_interval_kw()
            pa = PeakOffPeakAnalyser(profile)
            calc = tariff_calculator_class(
                pa, roof_area_sqm=row.get("estimated_area_sqm", 1000)
            )
            report = calc.annual_cost_report()
            rows.append(
                {
                    "name": row.get("name", ""),
                    "address": row.get("address", ""),
                    "area_sqm": row.get("estimated_area_sqm", 0),
                    "annual_kwh": report.total_kwh,
                    "annual_cost_aud": report.total_annual_cost,
                    "solar_kwp": report.installable_kwp,
                    "solar_saving_aud": report.solar_annual_saving,
                    "payback_years": report.solar_simple_payback_years,
                }
            )
        return pd.DataFrame(rows).sort_values("solar_saving_aud", ascending=False)
