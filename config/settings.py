"""
Central configuration for the Warehouse No Solar Sydney project.

No API keys are required for detection — Overpass (OSM) and NSW SIX Maps
are both free public services.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Free data sources — no key required
# ---------------------------------------------------------------------------

# OpenStreetMap Overpass API — warehouse/industrial building discovery
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Esri World Imagery XYZ tiles — free, no API key required
# Each tile is 256×256 px; we stitch a 3×3 grid per building
# Resolution at Sydney lat −33.9°, zoom 19: ~0.25 m/px (lon), ~0.30 m/px (lat)
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_TILE_ZOOM = 19
ESRI_TILE_GRID = 3        # NxN tile stitch per building (3 → 768×768 px, ~190m×230m)
ESRI_TILE_SIZE_PX = 256   # Esri standard tile dimension

# AEMO NEM data endpoints
AEMO_BASE_URL = "https://visualisations.aemo.com.au/aemo/apps/api/report"
AEMO_NEMWEB_URL = "https://www.nemweb.com.au/Reports/CURRENT"
NEM_REGION = "NSW1"

# Greater Sydney bounding box (lat/lng)
SYDNEY_BOUNDS = {
    "north": -33.40,
    "south": -34.20,
    "east": 151.35,
    "west": 150.50,
}

# Warehouse detection thresholds
MIN_WAREHOUSE_AREA_SQM = 500
SOLAR_PIXEL_THRESHOLD = 0.08     # fraction of dark-blue/navy pixels indicating panels
WAREHOUSE_ASPECT_RATIO_MIN = 1.2  # width/height ratio typical of industrial roofs

# Ausgrid network tariff (EA305 General Large Business, 2024–25 FY)
# Source: Ausgrid Network Price List 2024-25
AUSGRID_TARIFFS = {
    "peak_rate_kwh": 0.1234,       # $/kWh  07:00–22:00 weekdays (Nov–Mar)
    "shoulder_rate_kwh": 0.0712,   # $/kWh  07:00–22:00 weekdays (Apr–Oct) + Sat 07:00–22:00
    "offpeak_rate_kwh": 0.0348,    # $/kWh  all other times
    "demand_charge_kw": 12.80,     # $/kW/month  metered monthly maximum demand
    "daily_supply_charge": 3.85,   # $/day
}

# Retail energy cost addition (average NSW retailer margin, 2024)
RETAIL_ENERGY_MARGIN_KWH = 0.085  # $/kWh on top of network

# Solar potential benchmarks (NREL/CEC data adjusted for Sydney latitude −33.9°)
SOLAR_KWH_PER_KWP_PER_DAY = 4.2   # Sydney average daily yield
SOLAR_COST_PER_KWP = 1_050         # $ installed (commercial, >100kWp system)
PANEL_KWP_PER_SQM_ROOF = 0.170     # ~170W/m² for modern commercial panels (60% coverage)

# NEM data interval
NEM_INTERVAL_MINUTES = 30

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
