"""
AEMO NEM interval data loader for NSW region.

Downloads and parses 30-minute trading interval data from AEMO NEMWeb.
Data format reference: https://www.nemweb.com.au/Reports/CURRENT/
"""
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from config.settings import (
    AEMO_NEMWEB_URL,
    NEM_REGION,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class NEMLoader:
    """Download and parse AEMO NEM 30-minute interval demand data for NSW."""

    # AEMO DISPATCHREGIONSUM gives 5-min dispatch; TRADINGREGIONSUM gives 30-min trading
    TRADING_REPORT = "TRADINGREGIONSUM"
    DISPATCH_REPORT = "DISPATCHREGIONSUM"

    COLUMN_MAP = {
        "SETTLEMENTDATE": "timestamp",
        "REGIONID": "region",
        "TOTALDEMAND": "total_demand_mw",
        "SCHEDULEDGENERATION": "scheduled_gen_mw",
        "NETINTERCHANGE": "net_interchange_mw",
        "DEMAND_AND_NONSCHEDGEN": "demand_and_nonschedgen_mw",
    }

    def __init__(
        self,
        region: str = NEM_REGION,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.region = region
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_date_range(
        self,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame of 30-minute interval demand for *start* to *end* inclusive."""
        frames = []
        current = start
        with tqdm(total=(end - start).days + 1, desc="Loading NEM data") as pbar:
            while current <= end:
                df = self._load_day(current, use_cache=use_cache)
                if df is not None:
                    frames.append(df)
                current += timedelta(days=1)
                pbar.update(1)

        if not frames:
            logger.warning("No NEM data retrieved for %s – %s", start, end)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._clean(combined)
        return combined

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load a locally stored AEMO-formatted CSV (or the bundled sample)."""
        raw = pd.read_csv(path, skiprows=1, header=0)
        raw = raw[raw["I"] == "D"].copy()  # keep data rows only
        raw.columns = [c.strip() for c in raw.columns]
        return self._clean(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_day(self, day: date, use_cache: bool) -> Optional[pd.DataFrame]:
        filename = f"PUBLIC_DVD_{self.TRADING_REPORT}_{day:%Y%m%d}010000.zip"
        cache_path = self.cache_dir / filename

        if use_cache and cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return self._parse_zip(cache_path)

        url = f"{AEMO_NEMWEB_URL}/{self.TRADING_REPORT}/{filename}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Could not fetch %s: %s", url, exc)
            return None

        cache_path.write_bytes(response.content)
        return self._parse_zip(cache_path)

    def _parse_zip(self, zip_path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as fh:
                raw = pd.read_csv(fh, header=None, low_memory=False)

        # AEMO format: row[0] == "C" (comment), "I" (header), "D" (data)
        header_row = raw[raw[0] == "I"].iloc[0]
        data_rows = raw[raw[0] == "D"].copy()
        data_rows.columns = header_row.values
        return data_rows[data_rows["REGIONID"] == self.region].reset_index(drop=True)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.COLUMN_MAP)
        keep = [v for v in self.COLUMN_MAP.values() if v in df.columns]
        df = df[keep].copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in df.columns.difference(["timestamp", "region"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Convenience: resample to hourly or daily
    # ------------------------------------------------------------------

    @staticmethod
    def resample(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
        """Resample demand data to *freq* (e.g. 'h', 'D', 'W')."""
        numeric = df.select_dtypes(include="number")
        return numeric.resample(freq).mean()
