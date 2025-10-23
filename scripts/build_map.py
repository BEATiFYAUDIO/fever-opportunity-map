# scripts/build_map.py
import os, sys, json, statistics
from pathlib import Path
from typing import Dict, Any, List
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
OUT_GEOJSON = DOCS / "fever_events.geojson"  # city-level table for dashboards

# === Config ===
API_KEY = os.getenv("TM_API_KEY")
if not API_KEY:
    print("❌ ERROR: TM_API_KEY missing.", file=sys.stderr)
    sys.exit(1)

COUNTRIES = [s.strip() for s in os.getenv("TM_COUNTRIES", "US,CA").split(",") if s.strip()]
KEYWORD   = os.getenv("TM_KEYWORD", "")      # e.g. "hip hop" or artist name
SEGMENT   = os.getenv("TM_SEGMENT", "Music")
PAGES_MAX = int(os.getenv("TM_PAGES_MAX", "3"))
SIZE      = int(os.getenv("TM_PAGE_SIZE", "200"))
TOP_GENRE_COUNT = 8  # number of genres to expose in the dropdown (+ "All" + "Other")

# Percentile color scaling (so the map isn't all blue)
COLOR_PCT_LOW  = float(os.getenv("COLOR_PCT_LOW",  "20"))  # try 10–30
COLOR_PCT_HIGH = float(os.getenv("COLOR_PCT_HIGH", "95"))  # try 90–99

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
    """Flatten TM events to point dicts; capture city/state/country + genre & prices."""
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
            continue  # skip bad/missing coords

        # sanity filter to drop obviously invalid coords
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

        classifs = ev.get("classifications", []) or []
        gen = None
        if classifs:
            c0 = classifs[0]
            gen = (c0.get("genre") or {}).get("name")

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

# === Aggregation + Scoring ===
def aggregate_city_stats(points: List[Dict[str, Any]]) -> pd.DataFrame:
    groups = defaultdict(list)
    for p in points:
        key = f"{p['city']},{p['country']}"
        groups[key].append(p)

    rows = []
    for key, plist in groups.items():
        city, country = key.split(",", 1)

        # coordinate snapping: most common (lat,lng) to avoid drift
        coords = [(p["lat"], p["lng"]) for p in plist]
        (lat, lng), _ = Counter(coords).most_common(1)[0]

        # most common state/province (for Trends REGION match)
        states = [p.get("state") for p in plist if p.get("state")]
        state = Counter(states).most_common(1)[0][0] if states else ""

        event_count = len(plist)
        prices = [p["price_min"] or p["price_max"] for p in plist if p.get("price_min") or p.get("price_max")]
        avg_price = statistics.mean(prices) if prices else 0

        # (genres still shown in popups, but not used in scoring)
        genres = [p.get("genre") for p in plist if p.get("genre")]
        top_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        top_genres_str = ", ".join(top_genres) if top_genres else "—"

        rows.append({
            "city": city, "state": state, "country": country,
            "event_count": event_count,
            "avg_price": avg_price,
            "top_genres": top_genres_str,
            "lat": lat, "lng": lng
        })
    return pd.DataFrame(rows)

def enrich_with_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each city:
      1) Try Trends 'CITY' within country
      2) Else 'REGION' (state/province)
      3) Else country-level interest_over_time
    Use multiple keywords: 'live music', 'concerts', and the city's top genre.
    Take the max as final demand proxy (0–100).
    """
    if df.empty:
        return df.assign(search_interest=0.0)

    pytrends = TrendReq(hl="en-US", tz=360)
    df["search_interest"] = 0.0

    countries = df["country"].dropna().unique().tolist()

    def city_genre_kw(row):
        tg = row.get("top_genres") or ""
        return (tg.split(",")[0] or "").strip() or None

    for cc in countries:
        sub_idx = df["country"] == cc
        sub = df[sub_idx].copy()
        if sub.empty:
            continue

        # 1) CITY resolution
        try:
            pytrends.build_payload(["live music", "concerts"], geo=cc, timeframe="today 3-m")
            city_tbl = pytrends.interest_by_region(resolution="CITY", inc_low_vol=True)
        except Exception:
            city_tbl = pd.DataFrame()

        # 2) REGION resolution
        try:
            pytrends.build_payload(["live music", "concerts"], geo=cc, timeframe="today 3-m")
            region_tbl = pytrends.interest_by_region(resolution="REGION", inc_low_vol=True)
        except Exception:
            region_tbl = pd.DataFrame()

        # 3) Country-level TS
        try:
            pytrends.build_payload(["live music", "concerts"], geo=cc, timeframe="today 3-m")
            country_ts = pytrends.interest_over_time()
            country_val = float(country_ts[["live music", "concerts"]].mean(axis=1).tail(12).mean()) if not country_ts.empty else 0.0
        except Exception:
            country_val = 0.0

        updates = []
        for idx, row in sub.iterrows():
            city_name = row["city"]
            state_name = (row.get("state") or "").strip()
            genre_kw = city_genre_kw(row)

            city_val = 0.0
            if not city_tbl.empty and city_name in city_tbl.index:
                city_val = float(city_tbl.loc[city_name][["live music", "concerts"]].max())

            region_val = 0.0
            if state_name and not region_tbl.empty and state_name in region_tbl.index:
                region_val = float(region_tbl.loc[state_name][["live music", "concerts"]].max())

            genre_val = 0.0
            if genre_kw:
                try:
                    # try region first for finer grain
                    pytrends.build_payload([genre_kw], geo=cc, timeframe="today 3-m")
                    g_tbl = pytrends.interest_by_region(resolution="REGION", inc_low_vol=True)
                    if not g_tbl.empty and state_name in g_tbl.index:
                        genre_val = float(g_tbl.loc[state_name][genre_kw])
                    else:
                        g_ts = pytrends.interest_over_time()
                        genre_val = float(g_ts[genre_kw].tail(12).mean()) if not g_ts.empty else 0.0
                except Exception:
                    genre_val = 0.0

            final = max(city_val, region_val, country_val, genre_val)
            updates.append((idx, final))

        for idx, v in updates:
            df.loc[idx, "search_interest"] = v

    df["search_interest"] = df["search_interest"].clip(lower=0, upper=100)
    return df

# ---- Scoring (NO genre diversity) ----
def _robust_scale(s: pd.Series) -> pd.Series:
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) or 1.0
    z = (s - med) / iqr
    # squash with logistic so tails don't dominate
    return (1 / (1 + np.exp(-z))).clip(0, 1)

def compute_opportunity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified: only events, price, and demand — no genre factor."""
    if df.empty:
        return df.assign(opportunity_score=0.0)

    df = df.copy()
    df["event_term"] = np.log1p(df["event_count"].clip(lower=0))
    df["price_term"] = df["avg_price"].fillna(0)
    df["trend_term"] = df["search_interest"].fillna(0)

    df["event_s"] = _robust_scale(df["event_term"])
    df["price_s"] = _robust_scale(df["price_term"])
    df["trend_s"] = _robust_scale(df["trend_term"])

    # Opportunity: affordable + active + trending
    df["opportunity_score"] = (
        df["event_s"] * 0.40 +
        (1 - df["price_s"]) * 0.25 +
        df["trend_s"] * 0.35
    ) * 100

    # Final rounding
    df["opportunity_score"] = df["opportunity_score"].round(1)
    return df

# === Coloring helpers (percentile-scaled so map uses full palette) ===
def hsl_hotcold_scaled(score: float, vmin: float, vmax: float) -> str:
    # normalize score to 0..1 based on provided bounds
    if vmax <= vmin:
        t = 0.5
    else:
        t = (score - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # 0 → blue (240deg), 1 → red (0deg)
    return f"hsl({240 - (240 * t):.0f},80%,50%)"

# === Mapping ===
def write_html_map(points: List[Dict[str, Any]], df_city: pd.DataFrame, path: Path):
    if df_city.empty:
        raise SystemExit("No city data to plot.")

    # center
    m = folium.Map(location=[df_city["lat"].mean(), df_city["lng"].mean()],
                   zoom_start=3, tiles="CartoDB dark_matter")

    # robust percentile bounds for coloring (varied map)
    vmin = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_LOW))
    vmax = float(np.percentile(df_city["opportunity_score"], COLOR_PCT_HIGH))
    if vmax <= vmin:
        vmin, vmax = df_city["opportunity_score"].min(), df_city["opportunity_score"].max()

    # 1) Heatmap (by city score, scaled to 0..1 using vmin/vmax)
    HeatMap(
        [[r.lat, r.lng, max(0.0, min(1.0, (r.opportunity_score - vmin) / (vmax - vmin)))]
         for r in df_city.itertuples()],
        name="Opportunity Heatmap", radius=25, blur=20, min_opacity=0.4
    ).add_to(m)

    # 2) City bubbles (summary)
    fg_city = FeatureGroup(name="City Opportunity Scores", show=True)
    for r in df_city.itertuples():
        color = hsl_hotcold_scaled(r.opportunity_score, vmin, vmax)
        state_label = f", {r.state}" if isinstance(r.state, str) and r.state else ""
        html = f"""
        <div style='font-size:13px;'>
          <b>{r.city}{state_label}, {r.country}</b><br/>
          🔥 <b>Opportunity Score:</b> {r.opportunity_score:.1f}<br/>
          🎟️ Events: {r.event_count}<br/>
          💰 Avg Ticket Price: ${r.avg_price:.0f}<br/>
          📈 Search Interest: {r.search_interest:.0f}<br/>
          🎶 Top Genres: {r.top_genres}<br/>
        </div>
        """
        CircleMarker([r.lat, r.lng], radius=8, color=color, fillColor=color,
                     fill=True, fill_opacity=0.9)\
            .add_child(folium.Popup(html, max_width=320)).add_to(fg_city)
    fg_city.add_to(m)

    # 3) Top 10 Hot Markets overlay (numbered)
    top10 = df_city.sort_values("opportunity_score", ascending=False).head(10).reset_index(drop=True)
    fg_top = FeatureGroup(name="Top 10 Hot Markets", show=True)
    for idx, r in top10.iterrows():
        rank = idx + 1
        color = hsl_hotcold_scaled(r["opportunity_score"], vmin, vmax)
        state_label = f", {r['state']}" if isinstance(r["state"], str) and r["state"] else ""
        html = f"""
        <div style='font-size:13px;'>
          <b>#{rank} — {r['city']}{state_label}, {r['country']}</b><br/>
          Opportunity Score: {r['opportunity_score']:.1f}<br/>
          Events: {r['event_count']} · Avg ${r['avg_price']:.0f}<br/>
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

    # 4) Individual Venues (with genre filter)
    fg_venues = FeatureGroup(name="Individual Venues (Filtered)", show=False)
    m.add_child(fg_venues)

    # Build markers + collect JS handles for filtering
    venue_js_entries = []   # [(marker_var_name, genreBucket)]
    genre_counts = Counter([p["genre"] for p in points if p.get("genre")])
    top_genres = [g for g,_ in genre_counts.most_common(TOP_GENRE_COUNT)]
    def genre_bucket(g):
        if not g: return "Other"
        return g if g in top_genres else "Other"

    for p in points:
        # score: city average
        mask = (df_city["city"] == p["city"]) & (df_city["country"] == p["country"])
        score = float(df_city.loc[mask, "opportunity_score"].mean()) if mask.any() else 0.0
        color = hsl_hotcold_scaled(score, vmin, vmax)
        html = f"""
        <div style='font-size:12px;'>
          <b>{p['name']}</b><br/>
          {p['venue']} — {p['city']}, {p['country']}<br/>
          <i>{p.get('genre','')}</i><br/>
          <a href='{p['url']}' target='_blank'>Tickets</a><br/>
          Score: {score:.1f}
        </div>
        """
        mk = CircleMarker([p["lat"], p["lng"]], radius=4,
                          color=color, fillColor=color, fill=True, fill_opacity=0.85)
        mk.add_child(folium.Popup(html, max_width=300))
        mk.add_to(fg_venues)
        venue_js_entries.append((mk.get_name(), genre_bucket(p.get("genre"))))

    # Inject dropdown + filtering JS (robust: waits for markers, stops map clicks)
    fg_venues_js = fg_venues.get_name()
    options_html = "".join([f"<option value='{g}'>{g}</option>" for g in ["All"] + top_genres + ["Other"]])

    # Safely build the JS object mapping without backslashes in f-strings
    mapping_pairs = []
    for name, genre in venue_js_entries:
        safe_genre = genre.replace('"', '\\"')  # escape quotes manually
        mapping_pairs.append(f"'{name}':'{safe_genre}'")
    mapping_js = ",\n      ".join(mapping_pairs)

    markers_js = ",\n      ".join([f"'{name}': {name}" for name, _ in venue_js_entries])

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
      var GENRE_OF = {{
        {mapping_js}
      }};
      var VENUE_MARKER = {{
        {markers_js}
      }};
      var VENUES_GROUP = {fg_venues_js};

      function applyGenreFilter(genre) {{
        Object.keys(VENUE_MARKER).forEach(function(k) {{
          var m = VENUE_MARKER[k];
          var g = GENRE_OF[k] || "Other";
          var show = (genre === "All" || g === genre);
          if (show) {{
            if (!VENUES_GROUP.hasLayer(m)) {{ VENUES_GROUP.addLayer(m); }}
          }} else {{
            if (VENUES_GROUP.hasLayer(m)) {{ VENUES_GROUP.removeLayer(m); }}
          }}
        }});
      }}

      // Robust init: wait for Leaflet objects to exist
      (function initGenreFilter() {{
        var sel = document.getElementById('genreFilter');
        if (!sel || typeof VENUES_GROUP === 'undefined') {{ return setTimeout(initGenreFilter, 60); }}
        // ensure markers are defined
        var keys = Object.keys(VENUE_MARKER);
        for (var i=0;i<keys.length;i++) {{
          if (typeof VENUE_MARKER[keys[i]] === 'undefined') {{ return setTimeout(initGenreFilter, 60); }}
        }}
        // stop clicks from dragging the map
        L.DomEvent.disableClickPropagation(sel.parentElement);
        applyGenreFilter("All");
        sel.addEventListener('change', function() {{ applyGenreFilter(this.value); }});
      }})();
    </script>
    {{% endmacro %}}
    """
    ctl = MacroElement(); ctl._template = Template(control_html)
    m.get_root().add_child(ctl)

    # Legend with numeric bounds
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 50px;
        right: 30px;
        width: 220px;
        height: 120px;
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
          <span>{vmin:.1f}</span><span>→</span><span>{vmax:.1f}</span>
        </div>
        <hr style="border-color:rgba(255,255,255,0.3); margin:6px 0;">
        <b>Layers:</b><br>
        - Opportunity Heatmap<br>
        - City Scores<br>
        - Top 10 Hot Markets<br>
        - Individual Venues (Filtered)
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

if __name__ == "__main__":
    main()
