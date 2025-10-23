# scripts/build_map.py
import os, sys, json, statistics, math, time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

import requests
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl, Marker
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
from pytrends.request import TrendReq

# === Paths ===
WORK = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
DOCS = WORK / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS / "fever_market_opportunity_map.html"
OUT_GEOJSON = DOCS / "fever_events.geojson"
OUT_TRENDS_CSV = DOCS / "trends_debug.csv"

# === Config ===
API_KEY = os.getenv("TM_API_KEY")
if not API_KEY:
    print("❌ ERROR: TM_API_KEY missing.", file=sys.stderr)
    sys.exit(1)

COUNTRIES = [s.strip() for s in os.getenv("TM_COUNTRIES", "US,CA").split(",") if s.strip()]
KEYWORD = os.getenv("TM_KEYWORD", "")
SEGMENT = os.getenv("TM_SEGMENT", "Music")
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "3"))
SIZE = int(os.getenv("TM_PAGE_SIZE", "200"))

TOP_GENRE_COUNT = 8

COLOR_PCT_LOW = float(os.getenv("COLOR_PCT_LOW", "25"))
COLOR_PCT_HIGH = float(os.getenv("COLOR_PCT_HIGH", "90"))

TRENDS_MODE = os.getenv("TRENDS_MODE", "country").lower()
TRENDS_SLEEP_MS = int(os.getenv("TRENDS_SLEEP_MS", "400"))
TRENDS_TERMS = [t.strip() for t in os.getenv("TRENDS_TERMS", "live music,concerts").split(",") if t.strip()]

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
    if SEGMENT: base["segmentName"] = SEGMENT
    if KEYWORD: base["keyword"] = KEYWORD
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
        if not venues:
            continue
        v = venues[0]
        loc = v.get("location") or {}
        try:
            lat, lng = float(loc["latitude"]), float(loc["longitude"])
        except Exception:
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
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
        gen = None
        classifs = ev.get("classifications", []) or []
        if classifs:
            gen = (classifs[0].get("genre") or {}).get("name")
        pts.append({
            "name": name,
            "venue": v.get("name") or "",
            "city": (v.get("city") or {}).get("name") or "",
            "state": (v.get("state") or {}).get("name") or "",
            "stateCode": (v.get("state") or {}).get("stateCode") or "",
            "country": (v.get("country") or {}).get("countryCode") or "",
            "genre": gen or "Other",
            "price_min": price_min,
            "price_max": price_max,
            "currency": currency,
            "url": ev.get("url") or "",
            "lat": lat, "lng": lng,
        })
    return pts

# === Aggregation ===
def _country_median_prices(points: List[Dict[str, Any]]) -> Dict[str, float]:
    bins = defaultdict(list)
    for p in points:
        pm = p.get("price_min"); px = p.get("price_max")
        price = pm if pm is not None else px
        if price is not None and p.get("country"):
            bins[p["country"]].append(float(price))
    return {k: (statistics.median(v) if v else math.nan) for k, v in bins.items()}

def aggregate_city_stats(points: List[Dict[str, Any]]) -> pd.DataFrame:
    groups = defaultdict(list)
    for p in points:
        groups[f"{p['city']},{p['country']}"].append(p)
    country_median = _country_median_prices(points)
    rows = []
    for key, plist in groups.items():
        city, country = key.split(",", 1)
        coords = [(p["lat"], p["lng"]) for p in plist]
        (lat, lng), _ = Counter(coords).most_common(1)[0]
        states = [p.get("state") for p in plist if p.get("state")]
        state = Counter(states).most_common(1)[0][0] if states else ""
        event_count = len(plist)
        prices = [(p["price_min"] or p["price_max"]) for p in plist if (p["price_min"] or p["price_max"])]
        avg_price = float(statistics.mean(prices)) if prices else float(country_median.get(country, math.nan))
        genres = [p.get("genre") for p in plist if p.get("genre")]
        top_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        top_genres_str = ", ".join(top_genres) if top_genres else "—"
        rows.append({
            "city": city, "state": state, "country": country,
            "event_count": event_count, "avg_price": avg_price,
            "top_genres": top_genres_str, "lat": lat, "lng": lng,
        })
    return pd.DataFrame(rows)

# === Trends (country-level only) ===
def _trends_ts_mean(pytrends, terms, geo):
    try:
        pytrends.build_payload(terms, geo=geo, timeframe="today 3-m")
        ts = pytrends.interest_over_time()
        if ts is None or ts.empty:
            return 0.0
        vals = [float(ts[t].tail(12).mean()) for t in terms if t in ts.columns]
        return float(np.nanmax(vals)) if vals else 0.0
    except Exception:
        return 0.0

def enrich_with_trends(df):
    if df.empty or TRENDS_MODE == "off":
        return df.assign(search_interest=0.0)
    pytrends = TrendReq(hl="en-US", tz=360)
    df = df.copy(); df["search_interest"] = 0.0
    debug_rows = []
    for cc in df["country"].dropna().unique():
        terms = list(TRENDS_TERMS)
        val = _trends_ts_mean(pytrends, terms, cc)
        df.loc[df["country"] == cc, "search_interest"] = val
        debug_rows.append({"country": cc, "search_interest": val})
        time.sleep(TRENDS_SLEEP_MS / 1000.0)
    if df["search_interest"].max() <= 0.0:
        s = df["event_count"].astype(float)
        df["search_interest"] = (s - s.min()) / (s.max() - s.min()) * 100.0 if s.max() > s.min() else 50.0
    pd.DataFrame(debug_rows).to_csv(OUT_TRENDS_CSV, index=False)
    return df

# === Scoring ===
def _robust_scale(s):
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) or 1.0
    z = (s - med) / iqr
    return (1 / (1 + np.exp(-z))).clip(0, 1)

def compute_opportunity_scores(df):
    if df.empty: return df.assign(opportunity_score=0.0)
    df = df.copy()
    df["event_s"] = _robust_scale(np.log1p(df["event_count"]))
    df["price_s"] = _robust_scale(df["avg_price"].fillna(df["avg_price"].median()))
    df["trend_s"] = _robust_scale(df["search_interest"].fillna(0))
    df["opportunity_score"] = (
        df["event_s"] * 0.4 + (1 - df["price_s"]) * 0.25 + df["trend_s"] * 0.35
    ) * 100
    df["opportunity_score"] = df["opportunity_score"].round(1)
    return df

# === Mapping ===
def hsl_hotcold_scaled(score, vmin, vmax):
    if vmax <= vmin: t = 0.5
    else: t = (score - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return f"hsl({240 - 240*t:.0f},80%,50%)"

def write_html_map(points, df_city, path):
    if df_city.empty:
        raise SystemExit("No city data.")
    m = folium.Map(location=[df_city["lat"].mean(), df_city["lng"].mean()],
                   zoom_start=3, tiles="CartoDB dark_matter")
    vmin = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_LOW))
    vmax = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_HIGH))
    if vmax <= vmin: vmin, vmax = df_city["opportunity_score"].min(), df_city["opportunity_score"].max()
    HeatMap([[r.lat, r.lng, (r.opportunity_score - vmin)/(vmax-vmin)] for r in df_city.itertuples()],
            name="Opportunity Heatmap", radius=25, blur=20, min_opacity=0.4).add_to(m)
    fg_city = FeatureGroup(name="City Scores", show=True)
    for r in df_city.itertuples():
        color = hsl_hotcold_scaled(r.opportunity_score, vmin, vmax)
        CircleMarker([r.lat, r.lng], radius=8, color=color, fill=True, fill_opacity=0.9).add_to(fg_city)
    fg_city.add_to(m)

    # Individual venues w/ working filter
    fg_venues = FeatureGroup(name="Individual Venues", show=False)
    m.add_child(fg_venues)
    genre_counts = Counter([p["genre"] for p in points if p.get("genre")])
    top_genres = [g for g, _ in genre_counts.most_common(TOP_GENRE_COUNT)]
    def genre_bucket(g): return g if g in top_genres else "Other"
    for p in points:
        color = "orange"
        gtag = genre_bucket(p.get("genre", "Other"))
        mk = CircleMarker([p["lat"], p["lng"]], radius=4, color=color,
                          fill=True, fill_opacity=0.85, **{"feverGenre": gtag})
        mk.add_to(fg_venues)

    options_html = "".join([f"<option value='{g}'>{g}</option>" for g in ["All"] + top_genres + ["Other"]])
    control_html = f"""
    {{% macro html(this, kwargs) %}}
    <div id="genre-control" style="position: fixed; top: 80px; left: 10px; z-index: 9999;
        background: rgba(0,0,0,0.6); color: white; padding: 8px 10px; border-radius: 8px; font-size: 13px;">
      <label style="display:block; margin-bottom:4px;">Filter by Genre</label>
      <select id="genreFilter" style="width:180px; padding:4px;">{options_html}</select>
    </div>
    <script>
      var VENUES_GROUP = {fg_venues.get_name()};
      function applyGenreFilter(genre) {{
        if (!VENUES_GROUP || !VENUES_GROUP.eachLayer) return;
        VENUES_GROUP.eachLayer(function(m) {{
          try {{
            var g = (m && m.options && (m.options.feverGenre || m.options.genre)) || "Other";
            var show = (genre === "All" || g === genre);
            if (m._icon) m._icon.style.display = show ? "" : "none";
            if (m._shadow) m._shadow.style.display = show ? "" : "none";
            if (m.setStyle) m.setStyle({{opacity: show ? 1 : 0, fillOpacity: show ? 0.85 : 0}});
          }} catch (e) {{}}
        }});
      }}
      (function initGenreFilter() {{
        var sel = document.getElementById('genreFilter');
        if (!sel) return setTimeout(initGenreFilter, 60);
        L.DomEvent.disableClickPropagation(sel.parentElement);
        applyGenreFilter("All");
        sel.addEventListener('change', function() {{ applyGenreFilter(this.value); }});
      }})();
    </script>
    {{% endmacro %}}
    """
    ctl = MacroElement(); ctl._template = Template(control_html)
    m.get_root().add_child(ctl)

    LayerControl(collapsed=False).add_to(m)
    m.save(str(path))

def main():
    events = fetch_events()
    pts = to_points(events)
    df_city = aggregate_city_stats(pts)
    df_city = enrich_with_trends(df_city)
    df_city = compute_opportunity_scores(df_city)
    OUT_GEOJSON.write_text(df_city.to_json(orient="records"), encoding="utf-8")
    write_html_map(pts, df_city, OUT_HTML)
    print(f"✅ Map built at {OUT_HTML}")

if __name__ == "__main__":
    main()
