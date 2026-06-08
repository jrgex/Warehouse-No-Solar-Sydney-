"""
Warehouse detection and solar panel assessment — no API key required.

Pipeline:
  1. Query Overpass API (OpenStreetMap) for warehouse/industrial buildings
     in Greater Sydney with footprint area ≥ MIN_WAREHOUSE_AREA_SQM.
  2. Stitch a 3×3 grid of Esri World Imagery XYZ tiles (zoom 19, 768×768 px,
     ~190 m × 230 m coverage, ~0.25 m/px) centred on each building.
  3. Apply OpenCV heuristics to classify the rooftop:
       has_solar          — True/False
       solar_coverage_pct — % of roof pixels classified as PV panels
  4. Return a DataFrame of no-solar warehouses.

Free services used (no key required):
  • Overpass API         — https://overpass-api.de  (OpenStreetMap)
  • Esri World Imagery   — server.arcgisonline.com  (XYZ tiles, free/open)
"""
import logging
import math
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
    ESRI_TILE_GRID,
    ESRI_TILE_SIZE_PX,
    ESRI_TILE_URL,
    ESRI_TILE_ZOOM,
    MIN_WAREHOUSE_AREA_SQM,
    SOLAR_PIXEL_THRESHOLD,
    SYDNEY_BOUNDS,
)

logger = logging.getLogger(__name__)

# Metres per pixel at Esri zoom 19, Sydney latitude −33.9°
_EARTH_CIRC_M = 40_075_016.69
_TILE_METRES_LAT = _EARTH_CIRC_M / (2 ** ESRI_TILE_ZOOM)          # ~76.4 m/tile
_TILE_METRES_LNG = _TILE_METRES_LAT * math.cos(math.radians(33.9)) # ~63.5 m/tile
_METRES_PER_PX_LNG = _TILE_METRES_LNG / ESRI_TILE_SIZE_PX          # ~0.248 m/px
_METRES_PER_PX_LAT = _TILE_METRES_LAT / ESRI_TILE_SIZE_PX          # ~0.299 m/px

# Single lightweight tile query — sent once per ~5 km² grid cell
# Split into three simple statements to keep each request small and fast
_OVERPASS_TILE_QUERY = """\
[out:json][timeout:25];
way["building"~"^(warehouse|industrial|storage|distribution)$"]
  ({south},{west},{north},{east});
out body;>;out skel qt;
"""

# Overpass public endpoint with retry (exponential backoff on 429/504)
_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
_OVERPASS_MAX_RETRIES = 4
_OVERPASS_BACKOFF_BASE = 5  # seconds — doubles each retry

# Greater Sydney split into ~10 km grid cells (degrees)
# 10 km / 111 km·deg⁻¹ ≈ 0.090° lat;  10 km / 92.6 km·deg⁻¹ ≈ 0.108° lng
# Gives ~80 tiles vs 288 at 5 km — still small enough to avoid timeouts
_TILE_STEP_LAT = 0.090
_TILE_STEP_LNG = 0.108

# Polite delay between requests (seconds)
_WMS_REQUEST_DELAY = 0.4
_OVERPASS_TILE_DELAY = 3.0   # Overpass fair-use: stay well under 1 req/s


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

        The generator is consumed lazily — Overpass tile queries stop as soon
        as *max_results* buildings have been assessed, so --max 20 only fires
        as many tiles as needed to collect 20 candidates.
        """
        records: list[WarehouseRecord] = []
        assessed = 0
        for bldg in self._overpass_warehouses():
            if assessed >= max_results:
                break
            rec = self._assess_building(bldg)
            assessed += 1
            if rec is None:
                continue
            if not rec.has_solar and rec.estimated_area_sqm >= MIN_WAREHOUSE_AREA_SQM:
                records.append(rec)
                logger.info(
                    "[%d assessed] ✓ No solar: %s (%.0f m²)",
                    assessed, rec.name or rec.osm_id, rec.estimated_area_sqm,
                )
            else:
                logger.debug(
                    "[%d assessed] Skip %s — solar=%s area=%.0f m²",
                    assessed, rec.osm_id, rec.has_solar, rec.estimated_area_sqm,
                )
            time.sleep(_WMS_REQUEST_DELAY)

        logger.info("Done: %d buildings assessed, %d no-solar ≥500 m²", assessed, len(records))
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
    # Overpass / OSM  (tiled — one small request per ~5 km² cell)
    # ------------------------------------------------------------------

    def _overpass_warehouses(self) -> Iterator[dict]:
        """
        Tile Greater Sydney into ~5 km cells and query each cell separately.
        This keeps individual Overpass requests fast and avoids gateway timeouts
        on the public endpoint.
        """
        seen: set[str] = set()
        tiles = list(self._sydney_tiles())
        logger.info("Querying %d tiles via Overpass API…", len(tiles))

        for tile_idx, (s, w, n, e) in enumerate(tiles):
            query = _OVERPASS_TILE_QUERY.format(south=s, west=w, north=n, east=e)
            data = self._overpass_post(query, tile_idx + 1, len(tiles))
            if data is None:
                continue

            node_coords: dict[int, tuple[float, float]] = {
                el["id"]: (el["lat"], el["lon"])
                for el in data.get("elements", [])
                if el["type"] == "node"
            }
            for elem in data.get("elements", []):
                if elem["type"] != "way":
                    continue
                osm_id = str(elem["id"])
                if osm_id in seen:
                    continue
                seen.add(osm_id)

                nodes = elem.get("nodes", [])
                coords = [node_coords[nd] for nd in nodes if nd in node_coords]
                if len(coords) < 3:
                    continue

                area_sqm = self._polygon_area_sqm(coords)
                if area_sqm < MIN_WAREHOUSE_AREA_SQM:
                    continue

                lats = [c[0] for c in coords]
                lngs = [c[1] for c in coords]
                yield {
                    "osm_id": osm_id,
                    "centroid": (sum(lats) / len(lats), sum(lngs) / len(lngs)),
                    "area_sqm": area_sqm,
                    "tags": elem.get("tags", {}),
                }

            time.sleep(_OVERPASS_TILE_DELAY)

    @staticmethod
    def _sydney_tiles() -> Iterator[tuple[float, float, float, float]]:
        """Yield (south, west, north, east) bounding boxes covering Greater Sydney."""
        lat = SYDNEY_BOUNDS["south"]
        while lat < SYDNEY_BOUNDS["north"]:
            lng = SYDNEY_BOUNDS["west"]
            while lng < SYDNEY_BOUNDS["east"]:
                yield (
                    round(lat, 5),
                    round(lng, 5),
                    round(min(lat + _TILE_STEP_LAT, SYDNEY_BOUNDS["north"]), 5),
                    round(min(lng + _TILE_STEP_LNG, SYDNEY_BOUNDS["east"]), 5),
                )
                lng += _TILE_STEP_LNG
            lat += _TILE_STEP_LAT

    def _overpass_post(self, query: str, tile_n: int, total: int) -> Optional[dict]:
        """POST to Overpass with exponential backoff on 429/504."""
        delay = _OVERPASS_BACKOFF_BASE
        for attempt in range(1, _OVERPASS_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    _OVERPASS_ENDPOINT,
                    data={"data": query},
                    timeout=30,
                    headers={"User-Agent": "WarehouseNoSolarSydney/1.0"},
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", delay))
                    logger.warning(
                        "Tile %d/%d: rate-limited, waiting %ds (attempt %d/%d)",
                        tile_n, total, retry_after, attempt, _OVERPASS_MAX_RETRIES,
                    )
                    time.sleep(retry_after)
                    delay *= 2
                    continue
                resp.raise_for_status()
                logger.debug("Tile %d/%d OK", tile_n, total)
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Tile %d/%d attempt %d failed: %s — retrying in %ds",
                    tile_n, total, attempt, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
        logger.error("Tile %d/%d: all %d attempts failed, skipping", tile_n, total, _OVERPASS_MAX_RETRIES)
        return None

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

        image = self._fetch_esri_image(lat, lng)
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
    # Esri World Imagery — 3×3 tile stitch, no API key
    # ------------------------------------------------------------------

    def _fetch_esri_image(self, lat: float, lng: float) -> Optional[np.ndarray]:
        """
        Fetch and stitch a ESRI_TILE_GRID × ESRI_TILE_GRID grid of Esri World
        Imagery tiles centred on (lat, lng).  Result is a single BGR image at
        ~0.25 m/px covering ~190 m × 230 m.
        """
        cache_key = f"{lat:.6f}_{lng:.6f}"
        if self.image_cache_dir:
            cached = self.image_cache_dir / f"{cache_key}.png"
            if cached.exists():
                img = cv2.imread(str(cached))
                if img is not None:
                    return img

        cx, cy = self._latlon_to_tile(lat, lng, ESRI_TILE_ZOOM)
        half = ESRI_TILE_GRID // 2
        rows = []
        ok = True
        for dy in range(-half, half + 1):
            row_imgs = []
            for dx in range(-half, half + 1):
                tile = self._fetch_single_esri_tile(cx + dx, cy + dy, ESRI_TILE_ZOOM)
                if tile is None:
                    ok = False
                    break
                row_imgs.append(tile)
            if not ok:
                break
            rows.append(np.hstack(row_imgs))

        if not ok or not rows:
            return None

        stitched = np.vstack(rows)
        if self.image_cache_dir:
            cv2.imwrite(str(self.image_cache_dir / f"{cache_key}.png"), stitched)
        return stitched

    @staticmethod
    def _latlon_to_tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
        """Convert lat/lng to XYZ tile coordinates (standard Web Mercator)."""
        n = 2 ** zoom
        x = int((lng + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _fetch_single_esri_tile(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        url = ESRI_TILE_URL.format(z=z, y=y, x=x)
        try:
            resp = requests.get(
                url, timeout=15,
                headers={"User-Agent": "WarehouseNoSolarSydney/1.0 (+https://github.com/jrgex)"},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Esri tile z=%d x=%d y=%d failed: %s", z, x, y, exc)
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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
        pixel_area_sqm = _METRES_PER_PX_LNG * _METRES_PER_PX_LAT
        return round(cv2.countNonZero(roof_mask) * pixel_area_sqm, 1)

    def _detect_solar_panels(
        self, image: np.ndarray, roof_mask: np.ndarray
    ) -> tuple[bool, float]:
        """
        Detect solar panels from Esri World Imagery aerial tiles.
        PV panels appear as dark navy/blue-grey rectangular arrays on otherwise
        light-coloured industrial roofs (Colorbond, concrete, TPO).
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
