"""Download the REAL datasets used by JalRaksha into data/raw/.

Run once on a machine with internet (laptop or Google Colab):

    python fetch_data.py            # weather + elevation (needed)
    python fetch_data.py --roads    # also OpenStreetMap road graph for the routing tab (optional, slower)

All sources are free and need no API key:
  * Open-Meteo Forecast API            https://open-meteo.com/en/docs
  * Open-Meteo Historical Forecast API https://open-meteo.com/en/docs/historical-forecast-api
  * Open-Meteo Historical Weather API  https://open-meteo.com/en/docs/historical-weather-api  (ERA5-based)
  * Open-Meteo Elevation API           https://open-meteo.com/en/docs/elevation-api            (Copernicus DEM)
  * OpenStreetMap via osmnx (optional) https://www.openstreetmap.org
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from jalraksha import config as C

RAW = Path(__file__).parent / "data" / "raw"
TZ = "Asia/Kolkata"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ELEV = "https://api.open-meteo.com/v1/elevation"
ARCHIVE_LAG_DAYS = 6  # the ERA5-based archive trails real time by a few days


def get(url: str, params: dict, tries: int = 6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            print(f"  network error ({e}); retry {i + 1}/{tries}")
            time.sleep(3 * (i + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = 15 * (i + 1)
            print(f"  HTTP {r.status_code}: {r.text[:120]} -> waiting {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
    raise RuntimeError(f"Gave up on {url}")


def as_list(payload):
    return payload if isinstance(payload, list) else [payload]


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def tidy_daily(payload, names, models=None) -> pd.DataFrame:
    """Open-Meteo multi-location (+ optional multi-model) JSON -> long dataframe."""
    rows = []
    for name, item in zip(names, as_list(payload)):
        daily = item["daily"]
        dates = daily["time"]
        if models:
            for m in models:
                key = f"precipitation_sum_{m}"
                if key not in daily and len(models) == 1:
                    key = "precipitation_sum"
                if key not in daily:
                    continue
                for d, v in zip(dates, daily[key]):
                    rows.append((name, d, m, v))
        else:
            for d, v in zip(dates, daily["precipitation_sum"]):
                rows.append((name, d, None, v))
    cols = ["locality", "date", "model", "precip_mm"]
    df = pd.DataFrame(rows, columns=cols)
    return df if models else df.drop(columns="model")


def fetch_elevation(locs: pd.DataFrame) -> pd.DataFrame:
    pts = []
    for _, r in locs.iterrows():
        dlat = 1.5 / 111.0
        dlon = 1.5 / (111.32 * math.cos(math.radians(r.lat)))
        ring = [(r.lat + a * dlat, r.lon + b * dlon)
                for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
        pts.append([(r.lat, r.lon)] + ring)
    flat = [p for group in pts for p in group]
    elevs = []
    for ch in chunks(flat, 90):
        j = get(ELEV, {"latitude": ",".join(f"{a:.5f}" for a, _ in ch),
                       "longitude": ",".join(f"{b:.5f}" for _, b in ch)})
        elevs += j["elevation"]
    rows, k = [], 0
    for (_, r), group in zip(locs.iterrows(), pts):
        e = elevs[k:k + len(group)]
        k += len(group)
        rows.append({"locality": r.locality, "elev_m": e[0], "ring_mean_m": sum(e[1:]) / len(e[1:])})
    return pd.DataFrame(rows)


def fetch_history(locs, start, end):
    models = ",".join(C.MODELS)
    fc_parts, ob_parts = [], []
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    for grp in chunks(list(locs.itertuples()), 6):
        names = [g.locality for g in grp]
        lat = ",".join(str(g.lat) for g in grp)
        lon = ",".join(str(g.lon) for g in grp)
        for y in years:
            s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).date().isoformat()
            e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31")).date().isoformat()
            print(f"  history {names[0]}+{len(names) - 1} {s}..{e}")
            p = dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                     daily="precipitation_sum", models=models, timezone=TZ)
            fc_parts.append(tidy_daily(get(HIST_FC, p), names, C.MODELS))
            p2 = dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                      daily="precipitation_sum", timezone=TZ)
            ob_parts.append(tidy_daily(get(ARCHIVE, p2), names))
            time.sleep(1.0)
    return pd.concat(fc_parts, ignore_index=True), pd.concat(ob_parts, ignore_index=True)


def fetch_live(locs):
    parts = []
    for grp in chunks(list(locs.itertuples()), 8):
        names = [g.locality for g in grp]
        p = dict(latitude=",".join(str(g.lat) for g in grp), longitude=",".join(str(g.lon) for g in grp),
                 daily="precipitation_sum", models=",".join(C.MODELS), forecast_days=7,
                 past_days=0, timezone=TZ)
        parts.append(tidy_daily(get(FORECAST, p), names, C.MODELS))
    return pd.concat(parts, ignore_index=True)


def fetch_replay(locs):
    parts = []
    for grp in chunks(list(locs.itertuples()), 8):
        names = [g.locality for g in grp]
        p = dict(latitude=",".join(str(g.lat) for g in grp), longitude=",".join(str(g.lon) for g in grp),
                 start_date=C.REPLAY_START, end_date=C.REPLAY_END, daily="precipitation_sum", timezone=TZ)
        parts.append(tidy_daily(get(ARCHIVE, p), names))
    return pd.concat(parts, ignore_index=True)


def fetch_roads(out: Path):
    try:
        import osmnx as ox
    except ImportError:
        sys.exit("pip install osmnx   (needed only for --roads)")
    print("Downloading OSM major-road graph around Hyderabad (a few minutes)...")
    G = ox.graph_from_point((17.40, 78.47), dist=22000, network_type="drive", simplify=True,
                            custom_filter='["highway"~"motorway|trunk|primary|secondary"]')
    ox.save_graphml(G, out)
    print(f"  saved {out}  ({len(G.nodes)} nodes, {len(G.edges)} edges)")


def check():
    """10-second API sanity check: one locality, a few days. Prints what came back."""
    name, lat, lon = C.LOCALITIES[0]
    one = pd.DataFrame([(name, lat, lon)], columns=["locality", "lat", "lon"])
    print("elevation   :", fetch_elevation(one).round(1).to_dict("records"))
    live = fetch_live(one)
    print("live        :", live.groupby("model")["precip_mm"].agg(["count", "sum"]).round(1).to_dict("index"))
    p = dict(latitude=lat, longitude=lon, start_date="2023-08-01", end_date="2023-08-10", daily="precipitation_sum",
             models=",".join(C.MODELS), timezone=TZ)
    h = tidy_daily(get(HIST_FC, p), [name], C.MODELS)
    print("hist models :", h.groupby("model")["precip_mm"].agg(["count", "sum"]).round(1).to_dict("index"))
    o = tidy_daily(get(ARCHIVE, dict(latitude=lat, longitude=lon, start_date="2023-08-01", end_date="2023-08-10",
                                     daily="precipitation_sum", timezone=TZ)), [name])
    print("archive obs :", {"count": int(o.precip_mm.count()), "sum": round(float(o.precip_mm.sum()), 1)})
    print("OK - all four APIs reachable. Now run: python fetch_data.py --roads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="quick API sanity check, downloads almost nothing")
    ap.add_argument("--roads", action="store_true", help="also download OSM road graph")
    ap.add_argument("--start", default=C.HISTORY_START)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    if a.check:
        return check()
    RAW.mkdir(parents=True, exist_ok=True)
    end = a.end or (dt.date.today() - dt.timedelta(days=ARCHIVE_LAG_DAYS)).isoformat()

    locs = pd.DataFrame(C.LOCALITIES, columns=["locality", "lat", "lon"])
    locs.to_csv(RAW / "localities.csv", index=False)

    print("1/5 elevation + terrain ring ...")
    fetch_elevation(locs).to_csv(RAW / "elevation.csv", index=False)
    print("2/5 live 7-day multi-model forecast ...")
    fetch_live(locs).to_csv(RAW / "forecast_live.csv", index=False)
    print(f"3/5 history {a.start} .. {end} (model forecasts + ERA5 observations) ...")
    hist, obs = fetch_history(locs, a.start, end)
    hist.to_csv(RAW / "history_models.csv", index=False)
    obs.to_csv(RAW / "obs_era5.csv", index=False)
    print("4/5 October-2020 flood replay rainfall ...")
    fetch_replay(locs).to_csv(RAW / "replay_2020.csv", index=False)
    (RAW / "meta.json").write_text(json.dumps({
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "history_start": a.start, "history_end": end, "models": C.MODELS,
        "sources": ["Open-Meteo Forecast", "Open-Meteo Historical Forecast",
                    "Open-Meteo Historical Weather (ERA5)", "Open-Meteo Elevation (Copernicus DEM)"]},
        indent=2))
    if a.roads:
        print("5/5 roads ...")
        fetch_roads(RAW / "hyderabad_roads.graphml")
    else:
        print("5/5 roads skipped (run with --roads to enable the routing tab)")

    for f in ("history_models.csv", "obs_era5.csv"):
        d = pd.read_csv(RAW / f)
        print(f"  {f}: {len(d):,} rows, {d['precip_mm'].notna().mean():.0%} non-null")
    print("\nDone. Next: python run_pipeline.py")


if __name__ == "__main__":
    main()
