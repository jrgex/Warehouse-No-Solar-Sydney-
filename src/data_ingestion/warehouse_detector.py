"""
Warehouse detection and solar panel assessment using Google Maps imagery.

Pipeline:
  1. Query Google Maps Places API for industrial properties in Greater Sydney.
  2. Fetch satellite / aerial tiles via the Static Maps API.
  3. Apply computer-vision heuristics to classify each rooftop as:
       - has_solar  (True/False)
       - roof_area_sqm (estimated from pixel count + map scale)
  4. Filter to warehouses > MIN_WAREHOUSE_AREA_SQM with no solar.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd
import requests

from config.settings import (
    GOOGLE_MAPS_API_KEY,
    MIN_WAREHOUSE_AREA_SQM,
    SOLAR_PIXEL_THRESHOLD,
    SYDNEY_BOUNDS,
    WAREHOUSE_ASPECT_RATIO_MIN,
)

logger = logging.getLogger(__name__)

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Zoom level 20 → ~0.15 m/px in Sydney latitude; zoom 19 → ~0.30 m/px
ZOOM_LEVEL = 19
TILE_SIZE = 640  # pixels (max for free tier)
METRES_PER_PIXEL_Z19 = 0.298  # at latitude −33.9°


@dataclass
class WarehouseRecord:
    place_id: str
    name: str
    address: str
    lat: float
    lng: float
    estimated_area_sqm: float
    has_solar: bool
    solar_coverage_pct: float
    image_path: Optional[str] = None
    tags: list[str] = field(default_factory=list)


class WarehouseDetector:
    """Identify no-solar warehouses >500 m² in Greater Sydney."""

    def __init__(
        self,
        api_key: str = GOOGLE_MAPS_API_KEY,
        image_cache_dir: Optional[Path] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_MAPS_API_KEY is not set. "
                "Add it to your .env file before running detection."
            )
        self.api_key = api_key
        self.image_cache_dir = image_cache_dir
        if image_cache_dir:
            image_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_sydney(self, max_results: int = 200) -> pd.DataFrame:
        """Scan Greater Sydney for no-solar warehouses and return a DataFrame."""
        records = []
        for place in self._search_industrial_places(max_results=max_results):
            record = self._assess_property(place)
            if record and not record.has_solar and record.estimated_area_sqm >= MIN_WAREHOUSE_AREA_SQM:
                records.append(record)
                logger.info("Found: %s (%.0f m², no solar)", record.name, record.estimated_area_sqm)

        return self._to_dataframe(records)

    def assess_address(self, address: str) -> Optional[WarehouseRecord]:
        """Assess a single known warehouse address."""
        coords = self._geocode(address)
        if coords is None:
            logger.error("Could not geocode: %s", address)
            return None
        fake_place = {
            "place_id": address,
            "name": address,
            "vicinity": address,
            "geometry": {"location": {"lat": coords[0], "lng": coords[1]}},
        }
        return self._assess_property(fake_place)

    # ------------------------------------------------------------------
    # Places search
    # ------------------------------------------------------------------

    def _search_industrial_places(self, max_results: int) -> Iterator[dict]:
        """Yield place dicts from Places Nearby Search across Sydney grid."""
        # Divide Sydney bounding box into a coarse grid to stay within radius limits
        lat_steps = np.linspace(
            SYDNEY_BOUNDS["south"], SYDNEY_BOUNDS["north"], num=5
        )
        lng_steps = np.linspace(
            SYDNEY_BOUNDS["west"], SYDNEY_BOUNDS["east"], num=5
        )
        seen: set[str] = set()
        count = 0

        for lat in lat_steps:
            for lng in lng_steps:
                if count >= max_results:
                    return
                for place in self._places_page(lat, lng):
                    pid = place.get("place_id", "")
                    if pid in seen:
                        continue
                    seen.add(pid)
                    yield place
                    count += 1
                    if count >= max_results:
                        return

    def _places_page(self, lat: float, lng: float) -> list[dict]:
        params = {
            "location": f"{lat},{lng}",
            "radius": 8000,
            "type": "storage",
            "keyword": "warehouse logistics industrial",
            "key": self.api_key,
        }
        results = []
        while True:
            resp = requests.get(PLACES_NEARBY_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            page_token = data.get("next_page_token")
            if not page_token:
                break
            params = {"pagetoken": page_token, "key": self.api_key}
        return results

    # ------------------------------------------------------------------
    # Property assessment
    # ------------------------------------------------------------------

    def _assess_property(self, place: dict) -> Optional[WarehouseRecord]:
        loc = place["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]

        image = self._fetch_satellite_image(lat, lng)
        if image is None:
            return None

        roof_mask = self._extract_roof_mask(image)
        area_sqm = self._estimate_roof_area(roof_mask)
        has_solar, coverage_pct = self._detect_solar_panels(image, roof_mask)

        img_path: Optional[str] = None
        if self.image_cache_dir:
            pid = place.get("place_id", f"{lat}_{lng}").replace("/", "_")
            img_path = str(self.image_cache_dir / f"{pid}.png")
            cv2.imwrite(img_path, image)

        return WarehouseRecord(
            place_id=place.get("place_id", ""),
            name=place.get("name", "Unknown"),
            address=place.get("vicinity", ""),
            lat=lat,
            lng=lng,
            estimated_area_sqm=area_sqm,
            has_solar=has_solar,
            solar_coverage_pct=coverage_pct,
            image_path=img_path,
        )

    # ------------------------------------------------------------------
    # Satellite image fetching
    # ------------------------------------------------------------------

    def _fetch_satellite_image(self, lat: float, lng: float) -> Optional[np.ndarray]:
        cache_key = f"{lat:.6f}_{lng:.6f}_z{ZOOM_LEVEL}"
        if self.image_cache_dir:
            cached = self.image_cache_dir / f"{cache_key}.png"
            if cached.exists():
                return cv2.imread(str(cached))

        params = {
            "center": f"{lat},{lng}",
            "zoom": ZOOM_LEVEL,
            "size": f"{TILE_SIZE}x{TILE_SIZE}",
            "maptype": "satellite",
            "key": self.api_key,
        }
        try:
            resp = requests.get(STATIC_MAPS_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Image fetch failed for (%.6f, %.6f): %s", lat, lng, exc)
            return None

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if self.image_cache_dir and img is not None:
            cv2.imwrite(str(self.image_cache_dir / f"{cache_key}.png"), img)

        return img

    # ------------------------------------------------------------------
    # Computer-vision analysis
    # ------------------------------------------------------------------

    def _extract_roof_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask isolating large flat rooftop regions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold for light-coloured industrial roofs (metal, concrete, coated)
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Keep only contours with sufficiently rectangular aspect ratio
        mask = np.zeros_like(closed)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) == 0:
                continue
            ratio = max(w, h) / min(w, h)
            if ratio >= WAREHOUSE_ASPECT_RATIO_MIN:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    def _estimate_roof_area(self, roof_mask: np.ndarray) -> float:
        """Convert roof pixel count to square metres using map scale."""
        pixel_count = cv2.countNonZero(roof_mask)
        sqm = pixel_count * (METRES_PER_PIXEL_Z19 ** 2)
        return round(sqm, 1)

    def _detect_solar_panels(
        self, image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[bool, float]:
        """
        Classify whether solar panels are present on the roof.

        Solar panels in satellite imagery appear as dark navy/blue-black rectangular
        arrays with a slightly blue tint compared to surrounding roof material.
        """
        # Convert to HSV for colour segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Dark blue-black range: hue 100–140, low saturation possible, low value
        lower_solar = np.array([90, 20, 10])
        upper_solar = np.array([145, 255, 80])
        solar_mask = cv2.inRange(hsv, lower_solar, upper_solar)

        # Intersect with roof area only
        roof_only = cv2.bitwise_and(solar_mask, roof_mask)
        roof_pixels = cv2.countNonZero(roof_mask)
        if roof_pixels == 0:
            return False, 0.0

        solar_pixels = cv2.countNonZero(roof_only)
        coverage = solar_pixels / roof_pixels
        has_solar = coverage >= SOLAR_PIXEL_THRESHOLD
        return has_solar, round(coverage * 100, 2)

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------

    def _geocode(self, address: str) -> Optional[tuple[float, float]]:
        params = {"address": address, "key": self.api_key}
        try:
            resp = requests.get(GEOCODE_URL, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                loc = results[0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except requests.RequestException as exc:
            logger.error("Geocode error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(records: list[WarehouseRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=[
                    "place_id", "name", "address", "lat", "lng",
                    "estimated_area_sqm", "has_solar", "solar_coverage_pct",
                ]
            )
        return pd.DataFrame([vars(r) for r in records])
