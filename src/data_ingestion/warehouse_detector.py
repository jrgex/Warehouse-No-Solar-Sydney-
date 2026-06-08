"""
Warehouse detection and solar panel assessment — no API key required.

Pipeline:
  1. Query Overpass API (OpenStreetMap) for warehouse/industrial buildings
     in Greater Sydney with footprint area ≥ MIN_WAREHOUSE_AREA_SQM.
  2. Fetch a 640×640 aerial tile from the NSW SIX Maps WMS (free public service)
     for each building centroid.
  3. Apply OpenCV heuristics to classify the rooftop:
       has_solar          — True/False
       solar_coverage_pct — % of roof pixels classified as PV panels
  4. Return a DataFrame of no-solar warehouses.

Free services used:
  • Overpass API  — https://overpass-api.de (OpenStreetMap)
  • NSW SIX Maps  — https://maps.six.nsw.gov.au (NSW Government)
"""
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon

from config.settings import (
    MIN_WAREHOUSE_AREA_SQM,
    NSW_WMS_URL,
    OVERPASS_API_URL,
    SOLAR_PIXEL_THRESHOLD,
    SYDNEY_BOUNDS,
    WMS_COVERAGE_METRES,
    WMS_IMAGE_HEIGHT_PX,
    WMS_IMAGE_WIDTH_PX,
)

logger = logging.getLogger(__name__)

# Degrees of coverage for each WMS tile at Sydney latitude
_LAT_DEG_PER_TILE = WMS_COVERAGE_METRES / 111_000
_LNG_DEG_PER_TILE = WMS_COVERAGE_METRES / 92_600

# Overpass query — finds ways tagged as warehouse/industrial with a building tag
_OVERPASS_QUERY = """
[out:json][timeout:60];
(
  way["building"~"^(warehouse|industrial|storage|distribution)$"]
    ({south},{west},{north},{east});
  way["building"]["industrial"~"^(warehouse|distribution)$"]
    ({south},{west},{north},{east});
  way["building"]["landuse"="industrial"]
    ({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""

# Polite delay between WMS requests (seconds) to avoid rate-limit
_WMS_REQUEST_DELAY = 0.3


@dataclass
class WarehouseRecord:
    osm_id: str
    name: str
    address: str
    suburb: str
    lat: float
    lng: float
    estimated_area_sqm: float
    has_solar: bool
    solar_coverage_pct: float
    osm_tags: dict = field(default_factory=dict)
    image_path: Optional[str] = None


class WarehouseDetector:
    """
    Identify no-solar warehouses >500 m² in Greater Sydney.
    Uses Overpass API + NSW SIX Maps WMS — no API key required.
    """

    def __init__(self, image_cache_dir: Optional[Path] = None) -> None:
        self.image_cache_dir = image_cache_dir
        if image_cache_dir:
            image_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_sydney(self, max_results: int = 200) -> pd.DataFrame:
        """
        Scan Greater Sydney for no-solar warehouses.
        Returns a DataFrame sorted by estimated_area_sqm descending.
        """
        logger.info("Querying Overpass API for Sydney warehouses…")
        candidates = list(self._overpass_warehouses())
        logger.info("OSM returned %d candidate buildings", len(candidates))

        records: list[WarehouseRecord] = []
        for i, bldg in enumerate(candidates[:max_results]):
            rec = self._assess_building(bldg)
            if rec is None:
                continue
            if not rec.has_solar and rec.estimated_area_sqm >= MIN_WAREHOUSE_AREA_SQM:
                records.append(rec)
                logger.info(
                    "[%d/%d] ✓ No solar: %s (%.0f m²)",
                    i + 1, len(candidates), rec.name or rec.osm_id, rec.estimated_area_sqm,
                )
            else:
                logger.debug(
                    "[%d/%d] Skip: %s — solar=%s area=%.0f m²",
                    i + 1, len(candidates), rec.osm_id, rec.has_solar, rec.estimated_area_sqm,
                )
            time.sleep(_WMS_REQUEST_DELAY)

        df = self._to_dataframe(records)
        return df.sort_values("estimated_area_sqm", ascending=False).reset_index(drop=True)

    def assess_latlon(self, lat: float, lng: float, label: str = "") -> Optional[WarehouseRecord]:
        """Assess a single location by coordinates."""
        fake = {
            "osm_id": f"{lat:.5f}_{lng:.5f}",
            "centroid": (lat, lng),
            "area_sqm": 0.0,
            "tags": {"name": label},
        }
        return self._assess_building(fake)

    # ------------------------------------------------------------------
    # Overpass / OSM
    # ------------------------------------------------------------------

    def _overpass_warehouses(self) -> Iterator[dict]:
        """Yield building dicts from Overpass with centroid and area."""
        query = _OVERPASS_QUERY.format(**SYDNEY_BOUNDS)
        try:
            resp = requests.post(
                OVERPASS_API_URL,
                data={"data": query},
                timeout=90,
                headers={"User-Agent": "WarehouseNoSolarSydney/1.0"},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Overpass request failed: %s", exc)
            return

        data = resp.json()
        node_coords: dict[int, tuple[float, float]] = {
            n["id"]: (n["lat"], n["lon"])
            for n in data.get("elements", [])
            if n["type"] == "node"
        }

        for elem in data.get("elements", []):
            if elem["type"] != "way":
                continue
            nodes = elem.get("nodes", [])
            coords = [node_coords[n] for n in nodes if n in node_coords]
            if len(coords) < 3:
                continue

            area_sqm = self._polygon_area_sqm(coords)
            if area_sqm < MIN_WAREHOUSE_AREA_SQM:
                continue

            lats = [c[0] for c in coords]
            lngs = [c[1] for c in coords]
            centroid = (sum(lats) / len(lats), sum(lngs) / len(lngs))

            yield {
                "osm_id": str(elem["id"]),
                "centroid": centroid,
                "area_sqm": area_sqm,
                "tags": elem.get("tags", {}),
            }

    @staticmethod
    def _polygon_area_sqm(coords: list[tuple[float, float]]) -> float:
        """Approximate polygon area in m² using a local flat-earth projection."""
        lat0 = coords[0][0]
        metres_per_lat = 111_000.0
        metres_per_lng = 111_000.0 * np.cos(np.radians(lat0))
        xy = [(c[1] * metres_per_lng, c[0] * metres_per_lat) for c in coords]
        try:
            return Polygon(xy).area
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Building assessment
    # ------------------------------------------------------------------

    def _assess_building(self, bldg: dict) -> Optional[WarehouseRecord]:
        lat, lng = bldg["centroid"]
        tags = bldg.get("tags", {})

        image = self._fetch_wms_image(lat, lng)
        if image is None:
            return None

        roof_mask = self._extract_roof_mask(image)
        area_sqm = bldg["area_sqm"] if bldg["area_sqm"] > 0 else self._estimate_roof_area(roof_mask)
        has_solar, coverage_pct = self._detect_solar_panels(image, roof_mask)

        img_path: Optional[str] = None
        if self.image_cache_dir is not None:
            fname = f"{bldg['osm_id']}.png"
            img_path = str(self.image_cache_dir / fname)
            cv2.imwrite(img_path, image)

        name = tags.get("name") or tags.get("operator") or ""
        address = " ".join(filter(None, [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
        ]))
        suburb = tags.get("addr:suburb") or tags.get("addr:city") or ""

        return WarehouseRecord(
            osm_id=bldg["osm_id"],
            name=name,
            address=address,
            suburb=suburb,
            lat=lat,
            lng=lng,
            estimated_area_sqm=round(area_sqm, 1),
            has_solar=has_solar,
            solar_coverage_pct=coverage_pct,
            osm_tags=tags,
            image_path=img_path,
        )

    # ------------------------------------------------------------------
    # NSW SIX Maps WMS image fetching
    # ------------------------------------------------------------------

    def _fetch_wms_image(self, lat: float, lng: float) -> Optional[np.ndarray]:
        cache_key = f"{lat:.6f}_{lng:.6f}"
        if self.image_cache_dir:
            cached = self.image_cache_dir / f"{cache_key}.png"
            if cached.exists():
                return cv2.imread(str(cached))

        # Build WMS BBOX centred on the building
        half_lat = _LAT_DEG_PER_TILE / 2
        half_lng = _LNG_DEG_PER_TILE / 2
        bbox = f"{lng - half_lng},{lat - half_lat},{lng + half_lng},{lat + half_lat}"

        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": "0",
            "STYLES": "",
            "FORMAT": "image/jpeg",
            "SRS": "EPSG:4326",
            "BBOX": bbox,
            "WIDTH": WMS_IMAGE_WIDTH_PX,
            "HEIGHT": WMS_IMAGE_HEIGHT_PX,
        }
        try:
            resp = requests.get(NSW_WMS_URL, params=params, timeout=20,
                                headers={"User-Agent": "WarehouseNoSolarSydney/1.0"})
            resp.raise_for_status()
            # WMS returns an error XML if something goes wrong even on HTTP 200
            if b"<ServiceException" in resp.content[:200]:
                logger.warning("WMS service exception for (%.6f, %.6f)", lat, lng)
                return None
        except requests.RequestException as exc:
            logger.warning("WMS fetch failed for (%.6f, %.6f): %s", lat, lng, exc)
            return None

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not decode image for (%.6f, %.6f)", lat, lng)
            return None

        if self.image_cache_dir:
            cv2.imwrite(str(self.image_cache_dir / f"{cache_key}.png"), img)

        return img

    # ------------------------------------------------------------------
    # Computer-vision analysis
    # ------------------------------------------------------------------

    def _extract_roof_mask(self, image: np.ndarray) -> np.ndarray:
        """Binary mask of large flat rectangular roof regions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        mask = np.zeros_like(closed)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 400:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) == 0:
                continue
            if max(w, h) / min(w, h) >= 1.2:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    def _estimate_roof_area(self, roof_mask: np.ndarray) -> float:
        metres_per_pixel = WMS_COVERAGE_METRES / WMS_IMAGE_WIDTH_PX
        return round(cv2.countNonZero(roof_mask) * metres_per_pixel ** 2, 1)

    def _detect_solar_panels(
        self, image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[bool, float]:
        """
        Detect solar panels from aerial imagery.
        PV panels in NSW SIX Maps aerial images appear as dark navy/blue-grey
        rectangular arrays on otherwise light-coloured industrial roofs.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Dark blue-black HSV range for PV panels
        lower = np.array([90, 15, 10])
        upper = np.array([150, 255, 85])
        solar_mask = cv2.inRange(hsv, lower, upper)

        roof_only = cv2.bitwise_and(solar_mask, roof_mask)
        roof_pixels = cv2.countNonZero(roof_mask)
        if roof_pixels == 0:
            return False, 0.0

        coverage = cv2.countNonZero(roof_only) / roof_pixels
        return coverage >= SOLAR_PIXEL_THRESHOLD, round(coverage * 100, 2)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(records: list[WarehouseRecord]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=[
                "osm_id", "name", "address", "suburb", "lat", "lng",
                "estimated_area_sqm", "has_solar", "solar_coverage_pct",
            ])
        rows = []
        for r in records:
            row = {k: v for k, v in vars(r).items() if k != "osm_tags"}
            rows.append(row)
        return pd.DataFrame(rows)
