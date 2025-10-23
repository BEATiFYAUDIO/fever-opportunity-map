# scripts/build_map.py
import os, sys, json, statistics
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
from pytrends.request import TrendReq

# === Paths ===
WORK = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
DOCS = WORK / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS / "fever_market_opportunity_map.html"
OUT_GEOJSON = DOCS / "fever_events.geojson"

# === Config ===
API_KEY = os.getenv("TM_API_KEY")
if not API_KEY:
    print("❌ ERROR: TM_API_KEY missing.", file=sys.stderr)
    sys.exit(1)

COUNTRIES = [s.strip() for s in os.getenv("TM_COUNTRIES", "US,CA").split(",") if s.strip()]
KEYWORD   = os.getenv("TM_KEYWORD", "")
SEGMENT   = os.getenv("TM_SEGMENT", "Music")
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "3"))
SIZE      = int(os.getenv("TM_PAGE_SIZE", "200"))

# === Ticketmaster API ===
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

    all_events = []
    for cc in COUNTRIES:
        print(f"🌎 Fetching {SEGMENT} events for {cc}")
        page0 = tm_get(url, tm_params(base | {"countryCode": cc}, 0))
        total_pages = min(page0.get("page", {}).get("totalPages", 1), PAGES_MAX)
        all_events += page0.get("_embedded", {}).get("events", []) or []
        for p in range(1, total_pages):
            data = tm_get(url, tm_params(base | {"countryCode": cc}, p))
            all_events += data.get("_embedded", {}).get("events", []) or []
    print(f"✅ Retrieved {len(all_events)} events.")
    return all_events

def to_points(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pts = []
    for ev in events:
        venues = (ev.get("_embedded", {}) or {}).get("venues", []) or []
        if not venues: continue
        v = venues[0]; loc = v.get("location") or {}
        try:
            lat, lng = float(loc["latitude"]), float(loc["longitude"])
        except: continue
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
            "name": name, "date": when, "venue": v.get("name") or "",
            "city": (v.get("city") or {}).get("name") or "",
            "country": (v.get("country") or {}).get("countryCode") or "",
            "genre": gen, "price_min": price_min, "price_max": price_max,
            "currency": currency, "url": ev.get("url") or "",
            "lat": lat, "lng": lng
        })
    return pts

# === Aggregation + Scoring ===
def aggregate_city_stats(points: list[dict]) -> pd.DataFrame:
    groups = defaultdict(list)
    for p in points:
        key = f"{p['city']},{p['country']}"
        groups[key].append(p)

    rows = []
    for key, plist in groups.items():
        city, country = key.split(",", 1)
        event_count = len(plist)
        genres = [p.get("genre") for p in plist if p.get("genre")]
        top_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        genre_div = len(set(genres))
        prices = [p["price_min"] or p["price_max"] for p in plist if p.get("price_min") or p.get("price_max")]
        avg_price = statistics.mean(prices) if prices else 0
        lat = statistics.mean(p["lat"] for p in plist)
        lng = statistics.mean(p["lng"] for p in plist)
        rows.append({
            "city": city, "country": country, "event_count": event_count,
            "genre_div": genre_div, "avg_price": avg_price,
            "top_genres": ", ".join(top_genres), "lat": lat, "lng": lng
        })
    return pd.DataFrame(rows)

def enrich_with_trends(df: pd.DataFrame) -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=360)
    interests = []
    for city in df["city"].head(30):  # limit for runtime
        try:
            pytrends.build_payload(["live music"], geo="", timeframe="today 3-m")
            data = pytrends.interest_by_region(resolution="CITY", inc_low_vol=True)
            val = float(data.loc[city]["live music"]) if city in data.index else 0
        except Exception:
            val = 0
        interests.append(val)
    df["search_interest"] = interests + [0]*(len(df)-len(interests))
    return df

def compute_opportunity_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.assign(opportunity_score=0)

    def norm(series, invert=False):
        smin, smax = series.min(), series.max() or 1
        if smax == smin:
            return pd.Series([0.5]*len(series))
        v = (series - smin) / (smax - smin)
        return 1 - v if invert else v

    df["event_norm"] = norm(df["event_count"])
    df["genre_norm"] = norm(df["genre_div"])
    df["price_norm"] = norm(df["avg_price"], invert=True)
    df["trend_norm"] = norm(df["search_interest"])

    df["opportunity_score"] = (
        df["event_norm"]*0.35 +
        df["price_norm"]*0.15 +
        df["genre_norm"]*0.20 +
        df["trend_norm"]*0.30
    ) * 100
    return df

# === Mapping ===
def write_html_map(points: list[dict], df_city: pd.DataFrame, path: Path):
    if df_city.empty:
        raise SystemExit("No city data to plot.")

    avg_lat = df_city["lat"].mean()
    avg_lng = df_city["lng"].mean()
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=3, tiles="CartoDB dark_matter")

    # 1️⃣ Heatmap
    heat_data = df_city[["lat", "lng", "opportunity_score"]].dropna().values.tolist()
    HeatMap([[r[0], r[1], r[2] / 100] for r in heat_data],
            name="Opportunity Heatmap", radius=25, blur=20, min_opacity=0.4).add_to(m)

    # 2️⃣ City Bubbles
    fg_city = FeatureGroup(name="City Opportunity Scores", show=True)
    for row in df_city.itertuples():
        # blue (240°) → red (0°)
        color = f"hsl({240 - (240 * (row.opportunity_score / 100)):.0f},80%,50%)"
        html = f"""
        <div style='font-size:13px;'>
        <b>{row.city}, {row.country}</b><br/>
        🔥 <b>Opportunity Score:</b> {row.opportunity_score:.1f}<br/>
        🎟️ Events: {row.event_count}<br/>
        💰 Avg Ticket Price: ${row.avg_price:.0f}<br/>
        📈 Search Interest: {row.search_interest:.0f}<br/>
        🎶 Top Genres: {row.top_genres}<br/>
        </div>
        """
        CircleMarker(location=[row.lat, row.lng], radius=8, color=color,
                     fillColor=color, fill=True, fill_opacity=0.9)\
                     .add_child(folium.Popup(html, max_width=320)).add_to(fg_city)
    fg_city.add_to(m)

    # 3️⃣ Venues
    fg_venues = FeatureGroup(name="Individual Venues", show=False)
    for p in points:
        score = df_city.loc[df_city["city"] == p["city"], "opportunity_score"].mean() \
                 if p["city"] in df_city["city"].values else 0
        color = f"hsl({240 - (240 * (score / 100)):.0f},80%,50%)"
        html = f"""
        <div style='font-size:12px;'>
        <b>{p['name']}</b><br/>
        {p['venue']} — {p['city']} {p['country']}<br/>
        <i>{p.get('genre','')}</i><br/>
        <a href='{p['url']}' target='_blank'>Tickets</a><br/>
        Score: {score:.1f}
        </div>
        """
        CircleMarker(location=[p["lat"], p["lng"]], radius=4, color=color,
                     fillColor=color, fill=True, fill_opacity=0.8)\
                     .add_child(folium.Popup(html, max_width=300)).add_to(fg_venues)
    fg_venues.add_to(m)

    # 🔑 Legend
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        right: 30px;
        width: 180px;
        height: 110px;
        z-index:9999;
        font-size:13px;
        background: rgba(0,0,0,0.6);
        color: white;
        padding: 10px;
        border-radius: 10px;">
        <b>Opportunity Score</b><br>
        <div style="height:10px;
            background:linear-gradient(to right, blue, lightblue, yellow, orange, red);
            margin-top:5px; margin-bottom:5px;"></div>
        <div style="display:flex; justify-content:space-between;">
          <span>Cold</span><span>Warm</span><span>Hot</span>
        </div>
        <hr style="border-color:rgba(255,255,255,0.3); margin:6px 0;">
        <b>Layers:</b><br>
        - Opportunity Heatmap<br>
        - City Scores<br>
        - Venues
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    LayerControl(collapsed=False).add_to(m)
    m.save(str(path))

# === Main ===
def main():
    events = fetch_events()
    points = to_points(events)
    df_city = aggregate_city_stats(points)
    df_city = enrich_with_trends(df_city)
    df_city = compute_opportunity_scores(df_city)
    OUT_GEOJSON.write_text(df_city.to_json(orient="records"), encoding="utf-8")
    write_html_map(points, df_city, OUT_HTML)
    print(f"✅ Wrote {OUT_HTML}")

if __name__ == "__main__":
    main()
