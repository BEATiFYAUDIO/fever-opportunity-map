# scripts/build_map.py
import os, sys, json, math, statistics, time
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import folium
from folium import FeatureGroup, CircleMarker, LayerControl
import pandas as pd
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

# === Helpers ===
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
        if not venues: continue
        v = venues[0]; loc = v.get("location") or {}
        try: lat, lng = float(loc["latitude"]), float(loc["longitude"])
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

# === Fever Opportunity Model ===
def compute_scores(points: List[Dict[str, Any]]) -> Dict[str, float]:
    groups = defaultdict(list)
    for p in points:
        key = f"{p['city']},{p['country']}"
        groups[key].append(p)

    df = pd.DataFrame([
        {"city": p["city"], "country": p["country"],
         "genre": p["genre"],
         "price": p["price_min"] or p["price_max"] or None}
        for p in points if p.get("city")
    ])
    city_stats = []
    for key, plist in groups.items():
        city, country = key.split(",",1)
        event_count = len(plist)
        genres = {p["genre"] for p in plist if p.get("genre")}
        genre_div = len(genres)
        prices = [p["price_min"] or p["price_max"] for p in plist if p.get("price_min") or p.get("price_max")]
        avg_price = statistics.mean(prices) if prices else 0
        city_stats.append({"city":city,"country":country,
                           "event_count":event_count,
                           "genre_div":genre_div,
                           "avg_price":avg_price})
    df_city = pd.DataFrame(city_stats)
    if df_city.empty: return {}

    # 🔥 Google Trends demand (live music + top genre)
    pytrends = TrendReq(hl="en-US", tz=360)
    cities = df_city["city"].dropna().unique().tolist()
    interest = []
    for city in cities[:30]:  # limit to 30 for runtime
        try:
            pytrends.build_payload([f"{SEGMENT} concerts"], geo="", timeframe="today 3-m")
            data = pytrends.interest_by_region(resolution="CITY", inc_low_vol=True)
            if city in data.index:
                interest_val = float(data.loc[city][f"{SEGMENT} concerts"])
            else:
                interest_val = 0
        except Exception:
            interest_val = 0
        interest.append({"city": city, "search_interest": interest_val})
    df_trends = pd.DataFrame(interest)
    df_city = df_city.merge(df_trends, on="city", how="left").fillna({"search_interest":0})

    # Normalize all features 0–1
    for col in ["event_count","genre_div","avg_price","search_interest"]:
        if col not in df_city: continue
        vmin, vmax = df_city[col].min(), df_city[col].max() or 1
        if vmax == vmin: df_city[col+"_norm"] = 0.5
        else:
            if col == "avg_price":
                df_city[col+"_norm"] = 1 - ((df_city[col]-vmin)/(vmax-vmin))
            else:
                df_city[col+"_norm"] = (df_city[col]-vmin)/(vmax-vmin)

    # Weighted composite
    df_city["score"] = (
        df_city["event_count_norm"]*0.35 +
        df_city["avg_price_norm"]*0.15 +
        df_city["genre_div_norm"]*0.20 +
        df_city["search_interest_norm"]*0.30
    ) * 100
    return {f"{r.city},{r.country}": round(r.score,1) for r in df_city.itertuples()}

# === Output ===
def write_geojson(points: List[Dict[str, Any]], scores: Dict[str,float], path: Path):
    fc = {"type":"FeatureCollection","features":[]}
    for i,p in enumerate(points):
        key=f"{p['city']},{p['country']}"
        props={**p,"id":f"tm-{i}","opportunity_score":scores.get(key,0)}
        fc["features"].append({
            "type":"Feature",
            "properties":props,
            "geometry":{"type":"Point","coordinates":[p["lng"],p["lat"]]}
        })
    path.write_text(json.dumps(fc,ensure_ascii=False),encoding="utf-8")

def write_html_map(points: List[Dict[str, Any]], scores: Dict[str,float], path: Path):
    if not points: raise SystemExit("No events to plot.")
    avg_lat = sum(p["lat"] for p in points)/len(points)
    avg_lng = sum(p["lng"] for p in points)/len(points)
    m = folium.Map(location=[avg_lat,avg_lng], zoom_start=3, tiles="CartoDB dark_matter")
    fg = FeatureGroup(name="Venues",show=True)
    for p in points:
        key=f"{p['city']},{p['country']}"
        s=scores.get(key,0)
        color=f"hsl({120*(s/100):.0f},80%,50%)"
        html=f"""
        <b>{p['name']}</b><br/>
        {p['venue']} — {p['city']} {p['country']}<br/>
        <i>{p.get('genre','')}</i><br/>
        Opportunity Score: <b>{s}</b><br/>
        <a href='{p['url']}' target='_blank'>Tickets</a>
        """
        CircleMarker(location=[p["lat"],p["lng"]],radius=6,
            color=color,fillColor=color,fill=True,fill_opacity=0.9
        ).add_child(folium.Popup(html,max_width=300)).add_to(fg)
    fg.add_to(m); LayerControl(collapsed=True).add_to(m)
    m.save(str(path))

# === Main ===
def main():
    events = fetch_events()
    points = to_points(events)
    scores = compute_scores(points)
    write_geojson(points,scores,OUT_GEOJSON)
    write_html_map(points,scores,OUT_HTML)
    print(f"✅ Wrote {OUT_GEOJSON}")
    print(f"✅ Wrote {OUT_HTML}")

if __name__ == "__main__":
    main()

