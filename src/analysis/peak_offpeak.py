"""
Peak / off-peak / shoulder consumption analysis for Sydney warehouses.

Applies Ausgrid's time-of-use (TOU) period definitions to NEM interval data
or synthetic load profiles to break consumption into tariff bands.

Ausgrid EA305 TOU periods (2024-25):
  Peak:     07:00–22:00 weekdays, November–March
  Shoulder: 07:00–22:00 weekdays, April–October  |  07:00–22:00 Saturday
  Off-peak: All other times (22:00–07:00 every day, all Sunday, public holidays)

Reference: Ausgrid Network Price List 2024-25, Table 4.
"""
import logging
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# NSW public holidays 2024 (abridged – extend as needed)
NSW_PUBLIC_HOLIDAYS_2024 = pd.to_datetime(
    [
        "2024-01-01",  # New Year's Day
        "2024-01-26",  # Australia Day
        "2024-03-29",  # Good Friday
        "2024-04-01",  # Easter Monday
        "2024-04-25",  # Anzac Day
        "2024-06-10",  # King's Birthday
        "2024-08-05",  # Bank Holiday
        "2024-10-07",  # Labour Day
        "2024-12-25",  # Christmas Day
        "2024-12-26",  # Boxing Day
    ]
)

PERIOD_LABELS = ["peak", "shoulder", "offpeak"]


class PeakOffPeakAnalyser:
    """Classify energy consumption intervals into Ausgrid TOU tariff bands."""

    def __init__(
        self,
        load_profile: pd.DataFrame,
        demand_col: str = "demand_kw",
        interval_minutes: int = 30,
        public_holidays: pd.DatetimeIndex = NSW_PUBLIC_HOLIDAYS_2024,
    ) -> None:
        """
        Parameters
        ----------
        load_profile : DataFrame with DatetimeIndex (tz-aware or naïve Sydney local time)
                       and a demand column in kW.
        demand_col   : Name of the demand column.
        interval_minutes : Length of each interval for kWh conversion.
        """
        self.profile = load_profile.copy()
        self.demand_col = demand_col
        self.interval_hours = interval_minutes / 60
        self.public_holidays = public_holidays

        if not isinstance(self.profile.index, pd.DatetimeIndex):
            raise TypeError("load_profile must have a DatetimeIndex")

        self.profile["tou_period"] = self._classify_periods()

    # ------------------------------------------------------------------
    # Period classification
    # ------------------------------------------------------------------

    def _classify_periods(self) -> pd.Series:
        idx = self.profile.index
        # Localise naïve index to Sydney time
        if idx.tz is None:
            idx = idx.tz_localize("Australia/Sydney", ambiguous="NaT", nonexistent="NaT")

        is_holiday = idx.normalize().isin(self.public_holidays.normalize())
        is_weekday = (idx.dayofweek < 5) & ~is_holiday
        is_saturday = (idx.dayofweek == 5) & ~is_holiday
        hour = idx.hour
        month = idx.month
        is_summer = month.isin([11, 12, 1, 2, 3])
        is_winter = ~is_summer

        in_tou_window = (hour >= 7) & (hour < 22)

        peak = is_weekday & is_summer & in_tou_window
        shoulder = (is_weekday & is_winter & in_tou_window) | (is_saturday & in_tou_window)
        period = pd.Series("offpeak", index=self.profile.index)
        period[shoulder] = "shoulder"
        period[peak] = "peak"
        return period

    # ------------------------------------------------------------------
    # Consumption breakdown
    # ------------------------------------------------------------------

    def consumption_kwh(self) -> pd.Series:
        """kWh for each interval."""
        return self.profile[self.demand_col] * self.interval_hours

    def breakdown_by_period(self) -> pd.DataFrame:
        """Total kWh and average kW by TOU period."""
        kwh = self.consumption_kwh()
        df = pd.DataFrame(
            {
                "kwh": kwh,
                "tou_period": self.profile["tou_period"],
                "demand_kw": self.profile[self.demand_col],
            }
        )
        result = (
            df.groupby("tou_period")
            .agg(
                total_kwh=("kwh", "sum"),
                avg_demand_kw=("demand_kw", "mean"),
                peak_demand_kw=("demand_kw", "max"),
                interval_count=("kwh", "count"),
            )
            .reindex(PERIOD_LABELS)
        )
        result["pct_of_total"] = result["total_kwh"] / result["total_kwh"].sum() * 100
        return result.round(2)

    def monthly_breakdown(self) -> pd.DataFrame:
        """Monthly kWh by TOU period — useful for seasonal pattern analysis."""
        kwh = self.consumption_kwh().rename("kwh")
        df = pd.concat([kwh, self.profile["tou_period"]], axis=1)
        df["month"] = df.index.month
        pivot = df.pivot_table(
            index="month", columns="tou_period", values="kwh", aggfunc="sum"
        ).reindex(columns=PERIOD_LABELS, fill_value=0)
        pivot.index = pd.to_datetime(pivot.index, format="%m").strftime("%b")
        return pivot.round(1)

    def peak_demand_events(self, top_n: int = 10) -> pd.DataFrame:
        """Return the top-N highest demand intervals — relevant for demand charges."""
        series = self.profile[self.demand_col].nlargest(top_n)
        events = pd.DataFrame(
            {
                "demand_kw": series,
                "tou_period": self.profile.loc[series.index, "tou_period"],
            }
        )
        return events

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_daily_heatmap(
        self,
        figsize: tuple[int, int] = (14, 6),
        save_path: Union[str, None] = None,
    ) -> plt.Figure:
        """Heat map of average demand (kW) by hour-of-day and month."""
        df = self.profile[[self.demand_col]].copy()
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        pivot = df.pivot_table(
            values=self.demand_col, index="hour", columns="month", aggfunc="mean"
        )
        pivot.columns = pd.to_datetime(pivot.columns, format="%m").strftime("%b")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            cbar_kws={"label": "Average demand (kW)"},
            linewidths=0.3,
        )
        ax.set_title("Warehouse Demand Heatmap – Hour of Day × Month (Sydney)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour of day")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_tou_breakdown(
        self,
        figsize: tuple[int, int] = (8, 5),
        save_path: Union[str, None] = None,
    ) -> plt.Figure:
        """Bar chart of kWh by TOU period."""
        breakdown = self.breakdown_by_period()
        colours = {"peak": "#d62728", "shoulder": "#ff7f0e", "offpeak": "#1f77b4"}
        fig, ax = plt.subplots(figsize=figsize)
        breakdown["total_kwh"].plot(
            kind="bar",
            ax=ax,
            color=[colours.get(p, "grey") for p in breakdown.index],
            edgecolor="white",
        )
        ax.set_title("Annual Consumption by TOU Period (Ausgrid EA305)")
        ax.set_ylabel("kWh")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        for i, v in enumerate(breakdown["total_kwh"]):
            ax.text(i, v + v * 0.01, f"{v:,.0f}", ha="center", fontsize=9)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
