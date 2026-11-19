"""TEST FIXTURE ONLY - synthetic data in the same schema as data/raw, to exercise code offline.

Never ship this output as the dataset. Usage: python tests/make_fixture.py <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jalraksha import config as C


def make(out: Path, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    locs = pd.DataFrame(C.LOCALITIES, columns=["locality", "lat", "lon"])
    locs.to_csv(out / "localities.csv", index=False)
    elev = pd.DataFrame({
        "locality": locs.locality,
        "elev_m": 500 + rng.normal(0, 20, len(locs)),
        "ring_mean_m": 500 + rng.normal(0, 10, len(locs)),
    })
    elev.to_csv(out / "elevation.csv", index=False)
    dates = pd.date_range("2022-06-01", "2024-10-31")
    hist_rows, obs_rows = [], []
    bias = {"ecmwf_ifs025": 1.1, "gfs_seamless": 0.6, "icon_seamless": 1.4}
    noise = {"ecmwf_ifs025": 0.5, "gfs_seamless": 0.9, "icon_seamless": 0.7}
    for loc in locs.locality:
        season = np.where(dates.month.isin([6, 7, 8, 9, 10]), 1.0, 0.15)
        wet = rng.random(len(dates)) < 0.35 * season
        truth = np.where(wet, rng.gamma(0.8, 14, len(dates)), 0.0)
        for d, t in zip(dates, truth):
            obs_rows.append((loc, d.date().isoformat(), t))
        for m in C.MODELS:
            fc = np.clip(truth * bias[m] * np.exp(rng.normal(0, noise[m], len(dates)))
                         + np.where(rng.random(len(dates)) < 0.05, rng.gamma(1, 6, len(dates)), 0), 0, None)
            for d, v in zip(dates, fc):
                hist_rows.append((loc, d.date().isoformat(), m, v))
    pd.DataFrame(hist_rows, columns=["locality", "date", "model", "precip_mm"]).to_csv(out / "history_models.csv", index=False)
    pd.DataFrame(obs_rows, columns=["locality", "date", "precip_mm"]).to_csv(out / "obs_era5.csv", index=False)
    live_dates = pd.date_range("2024-11-05", periods=7)
    live = []
    for loc in locs.locality:
        base = rng.gamma(0.6, 25, 7)
        base[3] = 120 if loc in ("Chaderghat", "Malakpet", "Amberpet") else base[3]
        for m in C.MODELS:
            for d, b in zip(live_dates, base):
                live.append((loc, d.date().isoformat(), m, max(0, b * bias[m] * np.exp(rng.normal(0, .3)))))
    pd.DataFrame(live, columns=["locality", "date", "model", "precip_mm"]).to_csv(out / "forecast_live.csv", index=False)
    rd = pd.date_range(C.REPLAY_START, C.REPLAY_END)
    rep = [(loc, d.date().isoformat(), float(v)) for loc in locs.locality
           for d, v in zip(rd, rng.gamma(0.7, 40, len(rd)))]
    pd.DataFrame(rep, columns=["locality", "date", "precip_mm"]).to_csv(out / "replay_2020.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"fetched_at": "FIXTURE", "history_start": "2022-06-01",
                                               "history_end": "2024-10-31", "fixture": True}))


if __name__ == "__main__":
    make(Path(sys.argv[1]))
