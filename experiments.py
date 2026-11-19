"""Diagnose + tune the blend on 'tune' seasons, report on unseen 'holdout' seasons.

    python experiments.py

Prints raw models, equal mean, and 9 blend variants. Choose the variant with the lowest TUNE MAE,
then put its two numbers into jalraksha/config.py (RATIO_POWER, WEIGHT_POWER) and rerun run_pipeline.py.
"""
from pathlib import Path
import itertools
import pandas as pd
from jalraksha import blend
from jalraksha import config as C
from run_pipeline import wide_history

RAW = Path("data/raw")
locs = pd.read_csv(RAW / "localities.csv"); hist = pd.read_csv(RAW / "history_models.csv"); obs = pd.read_csv(RAW / "obs_era5.csv")
models = [m for m in C.MODELS if hist[hist.model == m].precip_mm.notna().any()]
wides = {l: wide_history(hist, obs, l, models) for l in locs.locality}
HOLD = C.HOLDOUT_FROM_YEAR


def evaluate(rp, wp):
    frames = []
    for loc, w in wides.items():
        if w["obs"].notna().sum() < 60:
            continue
        b = blend.blend_series(w, models, ratio_power=rp, weight_power=wp)
        b["locality"] = loc
        frames.append(b.reset_index().rename(columns={"index": "date"}))
    hb = pd.concat(frames, ignore_index=True); hb["date"] = pd.to_datetime(hb["date"])
    raw = [f"raw_{m}" for m in models]
    hb["ok"] = hb[raw].notna().all(axis=1) & hb["obs"].notna()
    hb["rank"] = hb[hb["ok"]].groupby("locality").cumcount()
    ev = hb[(hb["rank"] >= C.BIAS_WINDOW_DAYS) & hb["date"].dt.month.between(6, 10)]
    return ev


base_ev = evaluate(1.0, 1.0)
cols = [f"raw_{m}" for m in models] + ["mean_raw", "mean_corr"]
rows = []
for name, part in (("TUNE (before %d)" % HOLD, base_ev[base_ev.date.dt.year < HOLD]),
                   ("HOLDOUT (%d+)" % HOLD, base_ev[base_ev.date.dt.year >= HOLD])):
    sk = blend.skill(part, cols)
    for k, v in sk.items():
        rows.append(dict(split=name, forecast=k, **v))
res = {}
for rp, wp in itertools.product([0.0, 0.5, 1.0], [1.0, 2.0, 4.0]):
    ev = evaluate(rp, wp)
    for name, part in (("TUNE (before %d)" % HOLD, ev[ev.date.dt.year < HOLD]), ("HOLDOUT (%d+)" % HOLD, ev[ev.date.dt.year >= HOLD])):
        v = blend.skill(part, ["blend"]).get("blend")
        if v:
            rows.append(dict(split=name, forecast=f"blend ratio^{rp} weight^{wp}", **v))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for sp, g in df.groupby("split", sort=False):
    print(f"\n=== {sp}: {int(g['n'].iloc[0])} locality-days, {models} ===")
    print(g[["forecast", "mae", "rmse", "bias", "pod", "far", "csi"]].round(3).to_string(index=False))
tune = df[(df.split.str.startswith("TUNE")) & df.forecast.str.startswith("blend")]
if len(tune):
    best = tune.sort_values("mae").iloc[0]
    print("\nBest blend on TUNE seasons:", best["forecast"], " MAE", round(best["mae"], 3))
    h = df[(df.split.str.startswith("HOLD")) & (df.forecast == best["forecast"])]
    print("Its HOLDOUT result:\n", h[["forecast", "mae", "rmse", "bias", "pod", "far", "csi"]].round(3).to_string(index=False))
df.to_csv("data/processed/experiments.csv", index=False)
