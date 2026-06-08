# Warehouse No-Solar Sydney

Identifies and analyses warehouses in Greater Sydney (>500 m²) that have **no solar panels** on their rooftops, then estimates their electricity costs and solar savings potential using AEMO NEM interval data and Ausgrid network tariffs.

<p align="center">
  <a href="https://jrgex.github.io/Warehouse-No-Solar-Sydney-/data/warehouses_map.html">
    <img src="https://img.shields.io/badge/🗺️%20View%20Interactive%20Map-No--Solar%20Warehouses%20Sydney-blue?style=for-the-badge" alt="View Interactive Map" />
  </a>
</p>

---

## What it does

| Step | Module | Description |
|------|--------|-------------|
| 1 | `WarehouseDetector` | Queries Google Maps Places API for industrial properties in Greater Sydney, fetches satellite imagery, and classifies each rooftop: is it a warehouse? does it have solar? |
| 2 | `NEMLoader` | Downloads or parses AEMO NEM 30-minute trading interval data for the NSW region. |
| 3 | `BaselineDemandAnalyser` | Derives a baseline energy consumption profile — either from NEM data (rolling percentile method) or NABERS building intensity benchmarks. |
| 4 | `PeakOffPeakAnalyser` | Classifies every 30-minute interval into Ausgrid TOU bands: **peak**, **shoulder**, or **off-peak**. |
| 5 | `TariffCalculator` | Applies Ausgrid EA305 network tariffs (2024–25), retail margin, environmental levies and GST to produce a full annual cost estimate — plus a solar opportunity analysis. |

---

## Warehouse detection methodology

The `WarehouseDetector` uses two Google APIs:

- **Places Nearby Search** — finds properties tagged `storage`, `warehouse`, `logistics` across a 5×5 grid covering the Sydney bounding box.
- **Static Maps API** (satellite zoom 19) — retrieves a 640×640 px aerial image of each property.

Computer-vision heuristics (OpenCV) then:
1. **Roof mask** — isolates large flat rectangular light-coloured regions (Colorbond / concrete typical of industrial buildings) using adaptive thresholding and contour filtering on aspect ratio ≥ 1.2.
2. **Roof area** — converts pixel count to m² using the known map scale at zoom 19 (≈ 0.30 m/px at Sydney's latitude).
3. **Solar detection** — segments the HSV colour space for dark navy/blue-black pixels (solar PV panels) within the roof mask. If coverage ≥ 8%, the property is classified as `has_solar = True`.
4. **Filter** — only sites with `has_solar = False` **and** `estimated_area_sqm ≥ 500` are retained.

> **Note**: The CV heuristic is intentionally conservative. For production use, replace or augment it with a fine-tuned rooftop segmentation model (e.g. trained on the DeepSolar or SolarMapper datasets).

---

## Tariff structure — Ausgrid EA305 (2024–25)

| Period | Definition | Rate |
|--------|-----------|------|
| **Peak** | 07:00–22:00 weekdays, Nov–Mar | 12.34 c/kWh |
| **Shoulder** | 07:00–22:00 weekdays, Apr–Oct; 07:00–22:00 Saturday | 7.12 c/kWh |
| **Off-peak** | All other times | 3.48 c/kWh |
| **Demand charge** | Monthly maximum demand | $12.80/kW/month |
| **Daily supply** | | $3.85/day |

Retail margin, environmental levies (LRET, SRES, NSW ESC) and 10% GST are added on top.

---

## Solar opportunity model

| Parameter | Value |
|-----------|-------|
| Yield | 4.2 kWh/kWp/day (Sydney average, BOM/SolarEdge data) |
| Panel density | 170 Wp/m² (commercial monocrystalline) |
| Usable roof fraction | 60% (obstructions, orientation, setbacks) |
| Installed cost | $1,050/kWp (commercial >100 kWp, 2024 AU$) |

---

## Project structure

```
Warehouse-No-Solar-Sydney/
├── config/
│   └── settings.py          # All constants, tariffs, API keys (via .env)
├── data/
│   ├── sample_nem_nsw_2024.csv    # One day of NEM NSW trading data
│   └── sample_warehouses.csv     # 15 Sydney warehouses (no solar)
├── scripts/
│   └── run_analysis.py      # End-to-end pipeline runner
├── src/
│   ├── data_ingestion/
│   │   ├── nem_loader.py        # AEMO NEM data downloader / parser
│   │   └── warehouse_detector.py # Google Maps + CV rooftop analyser
│   ├── analysis/
│   │   ├── baseline_demand.py   # NABERS benchmark + rolling baseline
│   │   └── peak_offpeak.py      # Ausgrid TOU period classification
│   └── cost_estimation/
│       └── tariff_calculator.py # Full bill + solar savings model
├── tests/
│   └── test_analysis.py     # pytest unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/jrgex/Warehouse-No-Solar-Sydney-.git
cd Warehouse-No-Solar-Sydney-
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_MAPS_API_KEY
```

### 3. Run the sample analysis (no API key needed)

```bash
python scripts/run_analysis.py
```

### 4. Run live warehouse detection (requires Google Maps API key)

```bash
python scripts/run_analysis.py --detect
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Environment variables

Create a `.env` file in the project root:

```env
GOOGLE_MAPS_API_KEY=your_key_here
```

The key needs the following Google APIs enabled:
- Maps Static API
- Places API (Nearby Search)
- Geocoding API

---

## Data sources

| Source | Used for |
|--------|---------|
| [AEMO NEMWeb](https://www.nemweb.com.au/Reports/CURRENT/TRADINGREGIONSUM/) | NSW grid demand (30-min intervals) |
| [Ausgrid Network Price List 2024-25](https://www.ausgrid.com.au/Industry/Regulation-and-pricing/Network-prices) | TOU tariff rates |
| [NABERS Commercial Building Disclosure](https://www.energy.gov.au/government-priorities/energy-productivity-and-energy-efficiency/nabers) | Energy intensity benchmarks by building type |
| [BOM Solar Radiation Data](http://www.bom.gov.au/climate/averages/tables/cw_066062_Solar.shtml) | Sydney average daily solar yield |
| Google Maps Static + Places API | Satellite imagery and property search |

---

## Extending the project

- **Better solar detection**: Train a CNN on [DeepSolar](http://web.stanford.edu/group/deepsolar/home) or [SolarMapper](https://solarmapper.anl.gov/) labelled tiles and swap `warehouse_detector._detect_solar_panels()`.
- **Load monitoring**: Replace benchmark profiles with actual smart-meter interval data (AEMO MSATS or retailer data feeds).
- **Network tariff updates**: Update `config/settings.py` `AUSGRID_TARIFFS` dict each financial year.
- **Battery storage**: Extend `TariffCalculator` with a BESS dispatch model to capture demand charge reduction on top of solar savings.

---

## Licence

MIT
