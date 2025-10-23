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
KEYWORD   = os.getenv("TM_KEYWORD", "")
SEGMENT   = os.getenv("TM_SEGMENT", "Music")
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "3"))
SIZE      = int(os.getenv("TM_PAGE_SIZE", "200"))

TOP_GENRE_COUNT = 8

COLOR_PCT_LOW  = float(os.getenv("COLOR_PCT_LOW",  "25"))
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
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
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
            "name": name, "date": when, "venue": v.get("name") or "",
            "city": (v.get("city") or {}).get("name") or "",
            "state": (v.get("state") or {}).get("name") or "",
            "stateCode": (v.get("state") or {}).get("stateCode") or "",
            "country": (v.get("country") or {}).get("countryCode") or "",
            "genre": gen or "Other",
            "price_min": price_min, "price_max": price_max, "currency": currency,
            "url": ev.get("url") or "",
            "lat": lat, "lng": lng
        })
    return pts

# === Genre canonicalization (so Hip-Hop/Rap matches "Hip-Hop") ===
def canonical_genre(g: str | None) -> str:
    if not g: return "Other"
    s = g.lower()
    # Hip-Hop family
    if "hip" in s or "rap" in s: return "Hip-Hop"
    # EDM / Dance / Electronic
    if "edm" in s or "electronic" in s or "dance" in s: return "EDM"
    if "rock" in s: return "Rock"
    if "country" in s: return "Country"
    if "pop" in s: return "Pop"
    if "latin" in s or "reggaeton" in s: return "Latin"
    if "r&b" in s or "rnb" in s or "soul" in s: return "R&B"
    if "jazz" in s: return "Jazz"
    if "classical" in s or "symphony" in s: return "Classical"
    if "metal" in s: return "Metal"
    if "alternative" in s or "indie" in s: return "Alternative"
    if "folk" in s or "americana" in s: return "Folk"
    if "reggae" in s or "dancehall" in s: return "Reggae"
    if "comedy" in s: return "Comedy"
    return g  # default to original

# === Aggregation + Price Imputation ===
def _country_median_prices(points: List[Dict[str, Any]]) -> Dict[str, float]:
    bins: Dict[str, List[float]] = defaultdict(list)
    for p in points:
        pm = p.get("price_min"); px = p.get("price_max")
        price = pm if pm is not None else px
        if price is not None and p.get("country"):
            bins[p["country"]].append(float(price))
    return {k: (statistics.median(v) if v else math.nan) for k, v in bins.items()}

def aggregate_city_stats(points: List[Dict[str, Any]]) -> pd.DataFrame:
    groups = defaultdict(list)
    for p in points:
        key = f"{p['city']},{p['country']}"
        groups[key].append(p)

    country_median = _country_median_prices(points)

    rows = []
    for key, plist in groups.items():
        city, country = key.split(",", 1)

        coords = [(p["lat"], p["lng"]) for p in plist]
        (lat, lng), _ = Counter(coords).most_common(1)[0]

        states = [p.get("state") for p in plist if p.get("state")]
        state = Counter(states).most_common(1)[0][0] if states else ""
        state_codes = [p.get("stateCode") for p in plist if p.get("stateCode")]
        state_code = Counter(state_codes).most_common(1)[0][0] if state_codes else ""

        event_count = len(plist)
        prices = [
            (p["price_min"] if p["price_min"] is not None else p["price_max"])
            for p in plist if (p.get("price_min") is not None or p.get("price_max") is not None)
        ]
        if prices:
            avg_price = float(statistics.mean(prices))
        else:
            cm = country_median.get(country)
            avg_price = float(cm) if cm == cm else float("nan")

        # top genres (canonicalized for stability)
        genres = [canonical_genre(p.get("genre")) for p in plist if p.get("genre")]
        top_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        top_genres_str = ", ".join(top_genres) if top_genres else "—"

        rows.append({
            "city": city, "state": state, "stateCode": state_code, "country": country,
            "event_count": event_count, "avg_price": avg_price,
            "top_genres": top_genres_str, "lat": lat, "lng": lng
        })

    return pd.DataFrame(rows)

# === Trends (country-level + fallback) ===
def _trends_ts_mean(pytrends: TrendReq, terms: List[str], geo: str) -> float:
    try:
        pytrends.build_payload(terms, geo=geo, timeframe="today 3-m")
        ts = pytrends.interest_over_time()
        if ts is None or ts.empty:
            return 0.0
        vals = [float(ts[t].tail(12).mean()) for t in terms if t in ts.columns]
        return float(np.nanmax(vals)) if vals else 0.0
    except Exception:
        return 0.0

def enrich_with_trends(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or TRENDS_MODE == "off":
        return df.assign(search_interest=0.0)

    pytrends = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=0.4, timeout=(10, 30))
    df = df.copy()
    df["search_interest"] = 0.0

    debug_rows = []
    for cc in df["country"].dropna().unique():
        terms = list(TRENDS_TERMS)
        val = _trends_ts_mean(pytrends, terms, cc)
        df.loc[df["country"] == cc, "search_interest"] = float(max(0.0, min(100.0, val)))
        debug_rows.append({"country": cc, "terms": "|".join(terms), "search_interest": val})
        time.sleep(TRENDS_SLEEP_MS / 1000.0)

    if float(df["search_interest"].max()) <= 0.0:
        s = df["event_count"].astype(float)
        proxy = (s - s.min()) / (s.max() - s.min()) * 100.0 if s.max() > s.min() else pd.Series([50.0]*len(s), index=df.index)
        df["search_interest"] = proxy

    try:
        pd.DataFrame(debug_rows).to_csv(OUT_TRENDS_CSV, index=False)
    except Exception:
        pass

    return df

# === Scoring: events + price + demand ===
def _robust_scale(s: pd.Series) -> pd.Series:
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) or 1.0
    z = (s - med) / iqr
    return (1 / (1 + np.exp(-z))).clip(0, 1)

def compute_opportunity_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(opportunity_score=0.0)
    df = df.copy()
    df["event_term"] = np.log1p(df["event_count"].clip(lower=0))
    df["price_term"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df["price_term"] = df.groupby("country")["price_term"].transform(lambda s: s.fillna(s.median()))
    df["price_term"] = df["price_term"].fillna(df["price_term"].median())
    df["trend_term"] = df["search_interest"].fillna(0)
    df["event_s"] = _robust_scale(df["event_term"])
    df["price_s"] = _robust_scale(df["price_term"])
    df["trend_s"] = _robust_scale(df["trend_term"])
    df["opportunity_score"] = (df["event_s"]*0.40 + (1-df["price_s"])*0.25 + df["trend_s"]*0.35) * 100
    df["opportunity_score"] = df["opportunity_score"].round(1)
    return df

# === Color ===
def hsl_hotcold_scaled(score: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin: t = 0.5
    else: t = (score - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return f"hsl({240 - (240 * t):.0f},80%,50%)"

# === Map ===
def write_html_map(points: List[Dict[str, Any]], df_city: pd.DataFrame, path: Path):
    if df_city.empty:
        raise SystemExit("No city data to plot.")

    m = folium.Map(location=[df_city["lat"].mean(), df_city["lng"].mean()],
                   zoom_start=3, tiles="CartoDB dark_matter")

    vmin = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_LOW))
    vmax = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_HIGH))
    if vmax <= vmin:
        vmin, vmax = df_city["opportunity_score"].min(), df_city["opportunity_score"].max()

    # Heatmap
    heat = HeatMap(
        [[r.lat, r.lng, max(0.0, min(1.0, (r.opportunity_score - vmin) / (vmax - vmin)))]
         for r in df_city.itertuples()],
        name="Opportunity Heatmap", radius=25, blur=20, min_opacity=0.4
    )
    heat.add_to(m)

    # City bubbles
    fg_city = FeatureGroup(name="City Opportunity Scores", show=True)
    for r in df_city.itertuples():
        color = hsl_hotcold_scaled(r.opportunity_score, vmin, vmax)
        state_label = f", {r.state}" if isinstance(r.state, str) and r.state else ""
        price_label = "—" if (pd.isna(r.avg_price) or r.avg_price <= 0) else f"${r.avg_price:.0f}"
        html = f"""
        <div style='font-size:13px;'>
          <b>{r.city}{state_label}, {r.country}</b><br/>
          🔥 <b>Opportunity Score:</b> {r.opportunity_score:.1f}<br/>
          🎟️ Events: {r.event_count}<br/>
          💰 Avg Ticket Price: {price_label}<br/>
          📈 Search Interest: {r.search_interest:.0f}<br/>
          🎶 Top Genres: {r.top_genres}<br/>
        </div>
        """
        CircleMarker([r.lat, r.lng], radius=8, color=color, fillColor=color,
                     fill=True, fill_opacity=0.9)\
            .add_child(folium.Popup(html, max_width=320)).add_to(fg_city)
    fg_city.add_to(m)

    # Top 10 overlay
    top10 = df_city.sort_values("opportunity_score", ascending=False).head(10).reset_index(drop=True)
    fg_top = FeatureGroup(name="Top 10 Hot Markets", show=True)
    for idx, r in top10.iterrows():
        rank = idx + 1
        color = hsl_hotcold_scaled(r["opportunity_score"], vmin, vmax)
        state_label = f", {r['state']}" if isinstance(r["state"], str) and r["state"] else ""
        price_label = "—" if (pd.isna(r["avg_price"]) or r["avg_price"] <= 0) else f"${r['avg_price']:.0f}"
        html = f"""
        <div style='font-size:13px;'>
          <b>#{rank} — {r['city']}{state_label}, {r['country']}</b><br/>
          Opportunity Score: {r['opportunity_score']:.1f}<br/>
          Events: {r['event_count']} · Avg {price_label}<br/>
          Top Genres: {r['top_genres']}
        </div>
        """
        badge = folium.DivIcon(html=f"""
          <div style="
              background:{color};
              color:white; border-radius:16px; width:28px; height:28px;
              display:flex; align-items:center; justify-content:center;
              font-weight:700; font-size:12px; border:2px solid rgba(255,255,255,.85);
              box-shadow:0 0 8px rgba(0,0,0,.4);">{rank}</div>
        """, icon_size=(28,28), icon_anchor=(14,14))
        Marker([r["lat"], r["lng"]], icon=badge)\
            .add_child(folium.Popup(html, max_width=300)).add_to(fg_top)
    fg_top.add_to(m)

    # Individual Venues (with canonical genre tags)
    fg_venues = FeatureGroup(name="Individual Venues (Filtered)", show=False)
    m.add_child(fg_venues)

    # Canonicalize genres and build options
    canon_genres = [canonical_genre(p.get("genre")) for p in points if p.get("genre")]
    top_canon = [g for g, _ in Counter(canon_genres).most_common(TOP_GENRE_COUNT)]
    def bucket(g: str) -> str: return canonical_genre(g) if canonical_genre(g) in top_canon else "Other"

    # Precompute city->score for venue coloring
    key2score = {(r.city, r.country): float(r.opportunity_score) for r in df_city.itertuples()}

    for p in points:
        score = key2score.get((p["city"], p["country"]), 0.0)
        color = hsl_hotcold_scaled(score, vmin, vmax)
        gtag = bucket(p.get("genre"))

        html = f"""
        <div style='font-size:12px;'>
          <b>{p['name']}</b><br/>
          {p['venue']} — {p['city']}, {p['country']}<br/>
          <i>{canonical_genre(p.get('genre'))}</i><br/>
          <a href='{p['url']}' target='_blank'>Tickets</a><br/>
          Score: {score:.1f}
        </div>
        """
        CircleMarker([p["lat"], p["lng"]],
                     radius=4, color=color, fillColor=color, fill=True, fill_opacity=0.85,
                     **{"feverGenre": gtag}).add_child(folium.Popup(html, max_width=300)).add_to(fg_venues)

    # Dropdown + robust filtering (also auto-toggles layers for clarity)
    options_html = "".join([f"<option value='{g}'>{g}</option>" for g in ["All"] + top_canon + ["Other"]])
    control_html = f"""
    {{% macro html(this, kwargs) %}}
    <div id="genre-control" style="
        position: fixed; top: 80px; left: 10px; z-index: 9999;
        background: rgba(0,0,0,0.6); color: white; padding: 8px 10px;
        border-radius: 8px; font-size: 13px;">
      <label style="display:block; margin-bottom:4px;">Filter by Genre</label>
      <select id="genreFilter" style="width:180px; padding:4px;">
        {options_html}
      </select>
      <div style="margin-top:6px; opacity:.8;">(affects “Individual Venues” layer)</div>
    </div>
    <script>
      var MAP = {fg_venues.get_name()}._map;
      var VENUES = {fg_venues.get_name()};
      var LAYER_CITY = {fg_city.get_name()};
      var LAYER_TOP  = {fg_top.get_name()};
      var LAYER_HEAT = {heat.get_name()};

      function setLayerVisible(layer, show) {{
        if (!MAP) return;
        if (show) {{ if (!MAP.hasLayer(layer)) MAP.addLayer(layer); }}
        else      {{ if (MAP.hasLayer(layer)) MAP.removeLayer(layer); }}
      }}

      function applyGenreFilter(genre) {{
        if (!VENUES || !VENUES.eachLayer) return;

        // 1) Hide/show venue dots (DOM + vector)
        VENUES.eachLayer(function(m) {{
          try {{
            var g = (m && m.options && (m.options.feverGenre || m.options.genre)) || "Other";
            var show = (genre === "All" || g === genre);
            if (m._path)   m._path.style.display   = show ? "" : "none";
            if (m._icon)   m._icon.style.display   = show ? "" : "none";
            if (m._shadow) m._shadow.style.display = show ? "" : "none";
            if (m.setStyle) m.setStyle({{opacity: show ? 1 : 0, fillOpacity: show ? 0.85 : 0}});
            if (!show && m._popup && m._popup._container) m._popup._container.style.display = "none";
          }} catch (e) {{}}
        }});

        // 2) Auto-toggle other layers for clarity
        var filtered = (genre !== "All");
        setLayerVisible(VENUES, true);                 // ensure venues layer is on
        setLayerVisible(LAYER_CITY, !filtered);
        setLayerVisible(LAYER_TOP,  !filtered);
        setLayerVisible(LAYER_HEAT, !filtered);
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

    # Legend
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed; bottom: 50px; right: 30px; width: 220px; height: 120px;
        z-index:9999; font-size:13px; background: rgba(0,0,0,0.6);
        color: white; padding: 10px; border-radius: 10px;">
        <b>Opportunity Score</b><br>
        <div style="height:10px; background:linear-gradient(to right, blue, lightblue, yellow, orange, red);
            margin-top:5px; margin-bottom:5px;"></div>
        <div style="display:flex; justify-content:space-between;">
          <span>{vmin:.1f}</span><span>→</span><span>{vmax:.1f}</span>
        </div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement(); legend._template = Template(legend_html)
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
    print(f"✅ Wrote {OUT_GEOJSON}")
    if OUT_TRENDS_CSV.exists():
        print(f"🧪 Trends debug: {OUT_TRENDS_CSV}")

if __name__ == "__main__":
    main()
