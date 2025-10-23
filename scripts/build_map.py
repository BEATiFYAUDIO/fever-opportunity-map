# scripts/build_map.py
import os, sys, json, math, time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl

WORK = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
DOCS = WORK / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

OUT_HTML = DOCS / "fever_market_opportunity_map.html"
OUT_GEOJSON = DOCS / "fever_events.geojson"  # optional: for the live Leaflet page
API_KEY = os.getenv("TM_API_KEY")

if not API_KEY:
    print("ERROR: TM_API_KEY is not set. Add it as a repo secret.", file=sys.stderr)
    sys.exit(1)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def get_page(page: int):
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey": API_KEY,
        "classificationName": "Music",
        "size": 200,
        "sort": "date,asc",
        "page": page,
        # TODO: adjust filters for your use case, e.g.:
        # "countryCode": "US,CA",
        # "startDateTime": time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime()),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def collect_events(max_pages=5):
    events = []
    first = get_page(0)
    page_info = first.get("page", {}) or {}
    total_pages = page_info.get("totalPages", 1)
    embedded = first.get("_embedded", {}) or {}
    events.extend(embedded.get("events", []) or [])

    for p in range(1, min(total_pages, max_pages)):
        data = get_page(p)
        embedded = (data.get("_embedded", {}) or {})
        events.extend(embedded.get("events", []) or [])
    return events

def to_points(events):
    pts = []
    for ev in events:
        name = ev.get("name") or "Untitled"
        # Prefer localDate + localTime if present
        start = ev.get("dates", {}).get("start", {}) or {}
        when = " / ".join(filter(None, [start.get("localDate"), start.get("localTime")]))
        venues = (ev.get("_embedded", {}) or {}).get("venues", []) or []
        if not venues:
            continue
        v = venues[0]
        loc = v.get("location") or {}
        try:
            lat = float(loc["latitude"])
            lng = float(loc["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        city = v.get("city", {}).get("name") or ""
        country = v.get("country", {}).get("countryCode") or ""
        status = "upcoming"  # tune if you distinguish active/historical
        pts.append({
            "name": name, "date": when, "status": status,
            "lat": lat, "lng": lng, "city": city, "country": country
        })
    return pts

def write_geojson(points, path: Path):
    fc = {
        "type": "FeatureCollection",
        "features": []
    }
    for i, p in enumerate(points):
        fc["features"].append({
            "type": "Feature",
            "properties": {
                "id": f"tm-{i}",
                "name": p["name"],
                "time": p["date"],
                "status": p["status"],
                "city": p["city"],
                "country": p["country"],
            },
            "geometry": {"type": "Point", "coordinates": [p["lng"], p["lat"]]},
        })
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return path

def write_html_map(points, path: Path):
    if not points:
        raise SystemExit("No geocoded events to plot.")

    avg_lat = sum(p["lat"] for p in points)/len(points)
    avg_lng = sum(p["lng"] for p in points)/len(points)

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=3, tiles="CartoDB dark_matter")
    upcoming = FeatureGroup(name="Upcoming", show=True)

    for p in points:
        CircleMarker(
            location=[p["lat"], p["lng"]],
            radius=6, stroke=True, weight=1, fill=True, fill_opacity=0.9,
        ).add_child(folium.Popup(
            f"<b>{p['name']}</b><br>{p['date']}<br>{p['city']} {p['country']}"
        )).add_to(upcoming)

    upcoming.add_to(m)
    LayerControl(collapsed=True).add_to(m)
    m.save(str(path))

def main():
    events = collect_events(max_pages=5)
    points = to_points(events)

    # Write both artifacts: HTML for GH Pages, GeoJSON for the live Leaflet page
    write_html_map(points, OUT_HTML)
    write_geojson(points, OUT_GEOJSON)

    print("✅ Wrote:", OUT_HTML, "(", OUT_HTML.stat().st_size, "bytes )")
    print("✅ Wrote:", OUT_GEOJSON, "(", OUT_GEOJSON.stat().st_size, "bytes )")

if __name__ == "__main__":
    main()
