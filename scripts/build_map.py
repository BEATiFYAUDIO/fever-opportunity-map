# scripts/build_map.py
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl

WORK = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
DOCS = WORK / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

OUT_HTML = DOCS / "fever_market_opportunity_map.html"
OUT_GEOJSON = DOCS / "fever_events.geojson"

API_KEY = os.getenv("TM_API_KEY")
if not API_KEY:
    print("ERROR: TM_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

# ---- Filters (tune via repo Variables or inline) ----
# Examples:
#   TM_COUNTRIES=US,CA
#   TM_KEYWORD=rap OR hip hop
#   TM_MARKET_ID=27 (DMA/market, optional)
COUNTRIES = [s.strip() for s in os.getenv("TM_COUNTRIES", "US,CA").split(",") if s.strip()]
KEYWORD   = os.getenv("TM_KEYWORD", "")             # e.g., "rap OR hip hop" or artist name
SEGMENT   = os.getenv("TM_SEGMENT", "Music")        # keep as "Music" for music-only
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "5"))     # cap for runtime
SIZE      = int(os.getenv("TM_PAGE_SIZE", "200"))   # max 200

def tm_params(base: Dict[str, Any], page: int) -> Dict[str, Any]:
    p = {
        "apikey": API_KEY,
        "size": SIZE,
        "sort": "date,asc",
        "page": page,
    }
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
    if COUNTRIES:
        # Ticketmaster takes one countryCode at a time; we’ll loop
        pass

    all_events: List[Dict[str, Any]] = []
    country_list = COUNTRIES or [None]
    for cc in country_list:
        page0_params = tm_params(base | ({"countryCode": cc} if cc else {}), 0)
        data0 = tm_get(url, page0_params)
        page = data0.get("page", {}) or {}
        total_pages = min(page.get("totalPages", 1), PAGES_MAX)
        embedded = data0.get("_embedded", {}) or {}
        all_events.extend(embedded.get("events", []) or [])

        for p in range(1, total_pages):
            data = tm_get(url, tm_params(base | ({"countryCode": cc} if cc else {}), p))
            embedded = data.get("_embedded", {}) or {}
            all_events.extend(embedded.get("events", []) or [])
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
            lat = float(loc["latitude"]); lng = float(loc["longitude"])
        except Exception:
            continue

        # Metadata
        name = ev.get("name") or "Untitled"
        start = (ev.get("dates", {}) or {}).get("start", {}) or {}
        when  = " / ".join(filter(None, [start.get("localDate"), start.get("localTime")]))

        price_min = price_max = currency = None
        for pr in ev.get("priceRanges", []) or []:
            # pick the first advertised range
            price_min = pr.get("min", price_min)
            price_max = pr.get("max", price_max)
            currency  = pr.get("currency", currency)
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
    avg_lat = sum(p["lat"] for p in points)/len(points)
    avg_lng = sum(p["lng"] for p in points)/len(points)

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=3, tiles="CartoDB dark_matter")
    fg = FeatureGroup(name="Upcoming",
