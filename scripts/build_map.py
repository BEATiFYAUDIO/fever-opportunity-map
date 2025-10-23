# scripts/build_map.py
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl

# === Paths ===
WORK = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
DOCS = WORK / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS / "fever_market_opportunity_map.html"
OUT_GEOJSON = DOCS / "fever_events.geojson"

# === API Key ===
API_KEY = os.getenv("TM_API_KEY")
if not API_KEY:
    print("❌ ERROR: TM_API_KEY is not set in your repo secrets.", file=sys.stderr)
    sys.exit(1)

# === Optional filters ===
COUNTRIES = [s.strip() for s in os.getenv("TM_COUNTRIES", "US,CA").split(",") if s.strip()]
KEYWORD   = os.getenv("TM_KEYWORD", "")             # e.g. "rap OR hip hop"
SEGMENT   = os.getenv("TM_SEGMENT", "Music")        # usually "Music"
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "3"))     # limit for runtime
SIZE      = int(os.getenv("TM_PAGE_SIZE", "200"))

# === Ticketmaster helper ===
def tm_params(base: Dict[str, Any], page: int) -> Dict[str, Any]:
    p = {"apikey": API_KEY, "size": SIZE, "sort": "date,asc", "page": page}
    p.update(base)
    return p

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def tm_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_events() -> List[Dict[str, Any]]:
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    base = {}
    if SEGMENT:
        base["segmentName"] = SEGMENT
    if KEYWORD:
        base["keyword"] = KEYWORD

    all_events: List[Dict[str, Any]] = []
    for cc in COUNTRIES:
        print(f"🌎 Fetching {SEGMENT} events for {cc} ...")
        page0 = tm_get(url, tm_params(base | {"countryCode": cc}, 0))
        page_info = page0.get("page", {}) or {}
        total_pages = min(page_info.get("totalPages", 1), PAGES_MAX)
        all_events.extend(page0.get("_embedded", {}).get("events", []) or [])

        for p in range(1, total_pages):
            data = tm_get(url, tm_params(base | {"countryCode": cc}, p))
            all_events.extend(data.get("_embedded", {}).get("events", []) or [])
    print(f"✅ Retrieved {len(all_events)} events total.")
    return all_events

def to_points(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pts = []
    for ev in events:
        venues = (ev.get("_embedded", {}) or {}).get("venues", []) or []
        if not venues:
            continue
        v = venues[0]
        loc = v.get("location") or {}
        try:
            lat = float(loc["latitude"])
            lng = float(loc["longitude"])
        except Exception:
            continue

        name = ev.get("name") or "Untitled"
        start = (ev.get("dates", {}) or {}).get("start", {}) or {}
        when = " / ".join(filter(None, [start.get("localDate"), start.get("localTime")]))

        price_min = price_max = currency = None
        for pr in ev.get("priceRanges", []) or []:
            price_min = pr.get("min", price_min)
            price_max = pr.get("max", price_max)
            currency = pr.get("currency", currency)
            break

        classifs = ev.get("classifications", []) or []
        seg = gen = sub = None
        if classifs:
            c0 = classifs[0]
            seg = (c0.get("segment") or {}).get("name")
            gen = (c0.get("genre") or {}).get("name")
            sub = (c0.get("subGenre") or {}).get("name")

        pts.append({
            "name": name,
            "date": when,
            "venue": v.get("name") or "",
            "city": (v.get("city") or {}).get("name") or "",
            "country": (v.get("country") or {}).get("countryCode") or "",
            "segment": seg, "genre": gen, "subGenre": sub,
            "price_min": price_min, "price_max": price_max, "currency": currency,
            "url": ev.get("url") or "",
            "status": "upcoming",
            "lat": lat, "lng": lng
        })
    return pts

def write_geojson(points: List[Dict[str, Any]], path: Path) -> Path:
    fc = {"type": "FeatureCollection", "features": []}
    for i, p in enumerate(points):
        props = {k: p[k] for k in [
            "name","date","venue","city","country","segment","genre","subGenre",
            "price_min","price_max","currency","url","status"
        ]}
        fc["features"].append({
            "type": "Feature",
            "properties": {"id": f"tm-{i}", **props},
            "geometry": {"type": "Point", "coordinates": [p["lng"], p["lat"]]},
        })
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return path

def write_html_map(points: List[Dict[str, Any]], path: Path):
    if not points:
        raise SystemExit("No geocoded events to plot.")

    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lng = sum(p["lng"] for p in points) / len(points)
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=3, tiles="CartoDB dark_matter")
    fg = FeatureGroup(name="Upcoming", show=True)

    def fmt_price(p):
        if p["price_min"] is None and p["price_max"] is None:
            return ""
        if p["price_min"] is not None and p["price_max"] is not None:
            return f"{p['price_min']}–{p['price_max']} {p['currency'] or ''}"
        return f"{p['price_min'] or p['price_max']} {p['currency'] or ''}"

    for p in points:
        html = f"""
        <b>{p['name']}</b><br/>
        {p['date']}<br/>
        {p['venue']} — {p['city']} {p['country']}<br/>
        <i>{(p['segment'] or '')} {(p['genre'] or '')} {(p['subGenre'] or '')}</i><br/>
        {fmt_price(p)}<br/>
        <a href="{p['url']}" target="_blank" rel="noopener">Tickets</a>
        """
        CircleMarker(
            location=[p["lat"], p["lng"]],
            radius=6,
            stroke=True,
            weight=1,
            fill=True,
            fill_opacity=0.9,
        ).add_child(folium.Popup(html, max_width=320)).add_to(fg)

    fg.add_to(m)
    LayerControl(collapsed=True).add_to(m)
    m.save(str(path))

def main():
    events = fetch_events()
    points = to_points(events)
    write_geojson(points, OUT_GEOJSON)
    write_html_map(points, OUT_HTML)
    print(f"✅ Wrote {OUT_GEOJSON} ({OUT_GEOJSON.stat().st_size} bytes)")
    print(f"✅ Wrote {OUT_HTML} ({OUT_HTML.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
