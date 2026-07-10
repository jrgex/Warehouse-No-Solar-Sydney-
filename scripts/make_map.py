"""
Generate an interactive HTML map of no-solar warehouses in Greater Sydney.

Uses:
  • Nominatim (OSM) for reverse geocoding missing addresses — free, no key
  • Folium for the interactive Leaflet map
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import pandas as pd
import requests
from folium.plugins import MarkerCluster

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_HTML = DATA_DIR / "warehouses_map.html"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "WarehouseNoSolarSydney/1.0 (+https://github.com/jrgex)"}


def reverse_geocode(lat: float, lng: float) -> str:
    """Return a human-readable address from Nominatim, or coords on failure."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lng, "format": "json", "zoom": 18},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        addr = data.get("address", {})
        parts = [
            addr.get("house_number", ""),
            addr.get("road", ""),
            addr.get("suburb") or addr.get("town") or addr.get("city", ""),
            addr.get("state", ""),
        ]
        return ", ".join(p for p in parts if p)
    except Exception:
        return f"{lat:.5f}, {lng:.5f}"


def solar_payback(area_sqm: float) -> tuple[float, float, float]:
    """Return (annual_kwh, annual_cost_aud, payback_years) for a warehouse of given area."""
    from src.analysis.baseline_demand import BaselineDemandAnalyser
    from src.analysis.peak_offpeak import PeakOffPeakAnalyser
    from src.cost_estimation.tariff_calculator import TariffCalculator

    ba = BaselineDemandAnalyser(warehouse_area_sqm=area_sqm)
    profile = ba.benchmark_interval_kw()
    pa = PeakOffPeakAnalyser(profile)
    calc = TariffCalculator(pa, roof_area_sqm=area_sqm)
    r = calc.annual_cost_report()
    return r.total_kwh, r.total_annual_cost, r.solar_simple_payback_years or 0.0


def make_map(df: pd.DataFrame) -> folium.Map:
    centre_lat = df["lat"].mean()
    centre_lng = df["lng"].mean()

    m = folium.Map(
        location=[centre_lat, centre_lng],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # ── Legend ──────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-family:sans-serif;font-size:13px">
      <b>Sydney Warehouses — No Solar</b><br>
      <span style="color:#e05c00">●</span> &nbsp;Large (&gt;5 000 m²)<br>
      <span style="color:#e8a800">●</span> &nbsp;Medium (1 000–5 000 m²)<br>
      <span style="color:#2a7ae2">●</span> &nbsp;Small (500–1 000 m²)<br>
      <hr style="margin:6px 0">
      <small>Circle size ∝ roof area &nbsp;|&nbsp; Click for details</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    cluster = MarkerCluster(
        name="Warehouses",
        options={"maxClusterRadius": 50, "disableClusteringAtZoom": 14},
    ).add_to(m)

    for _, row in df.iterrows():
        area = row["estimated_area_sqm"]
        kwh, cost, payback = solar_payback(area)

        # Colour by size
        if area >= 5_000:
            colour = "#e05c00"
        elif area >= 1_000:
            colour = "#e8a800"
        else:
            colour = "#2a7ae2"

        radius = max(8, min(40, area ** 0.45 * 0.9))

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:240px;font-size:13px">
          <b style="font-size:14px">{row['name'] or 'Unnamed warehouse'}</b><br>
          <span style="color:#555">{row['address']}</span>
          <hr style="margin:6px 0">
          <table style="width:100%;border-collapse:collapse">
            <tr><td>Roof area</td><td align="right"><b>{area:,.0f} m²</b></td></tr>
            <tr><td>OSM ID</td><td align="right">{row['osm_id']}</td></tr>
            <tr style="color:#c00"><td>Solar panels</td><td align="right"><b>None detected</b></td></tr>
            <tr><td colspan=2 style="padding-top:6px;color:#888;font-size:11px">── Energy estimate ──</td></tr>
            <tr><td>Annual consumption</td><td align="right">{kwh:,.0f} kWh</td></tr>
            <tr><td>Est. annual bill</td><td align="right"><b>${cost:,.0f}</b></td></tr>
            <tr style="color:#2a7">
              <td>Solar payback</td><td align="right"><b>{payback:.1f} yr</b></td>
            </tr>
          </table>
          <div style="margin-top:8px;font-size:11px;color:#888">
            {row['lat']:.6f}, {row['lng']:.6f}
          </div>
        </div>"""

        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['name'] or 'Warehouse'} — {area:,.0f} m²",
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "detected_warehouses.csv")
    df["address"] = df["address"].astype(str).replace("nan", "")
    df["name"] = df["name"].astype(str).replace("nan", "")

    # Reverse geocode any rows with no address (NaN or blank)
    missing = df["address"].isna() | (df["address"].astype(str).str.strip() == "")
    print(f"Reverse geocoding {missing.sum()} addresses via Nominatim…")
    for i, row in df[missing].iterrows():
        if True:
            addr = reverse_geocode(row["lat"], row["lng"])
            df.at[i, "address"] = addr
            print(f"  {addr}")
            time.sleep(1.1)   # Nominatim rate limit: 1 req/s

    df.to_csv(DATA_DIR / "detected_warehouses.csv", index=False)
    print("Addresses saved.\n")

    print("Building map…")
    m = make_map(df)
    m.save(str(OUT_HTML))
    print(f"Map saved → {OUT_HTML}")
    print("\nOpen in Windows:\n  \\\\wsl.localhost\\Ubuntu-26.04\\home\\jr\\Warehouse-No-Solar-Sydney\\data\\warehouses_map.html")
