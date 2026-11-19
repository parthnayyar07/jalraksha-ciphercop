"""Turn data/raw/*.csv into data/processed/*: blended forecast, validation, risk tables.

    python run_pipeline.py [--raw data/raw] [--out data/processed]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jalraksha import blend, risk
from jalraksha import config as C


def wide_history(hist: pd.DataFrame, obs: pd.DataFrame, locality: str, models: list[str]) -> pd.DataFrame:
    h = hist[hist.locality == locality].pivot(index="date", columns="model", values="precip_mm")
    o = obs[obs.locality == locality].set_index("date")["precip_mm"].rename("obs")
    df = h.join(o, how="inner")
    df.index = pd.to_datetime(df.index)
    return df.sort_index().reindex(columns=models + ["obs"])


def main(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    locs = pd.read_csv(raw / "localities.csv")
    elev = pd.read_csv(raw / "elevation.csv")
    hist = pd.read_csv(raw / "history_models.csv")
    obs = pd.read_csv(raw / "obs_era5.csv")
    live = pd.read_csv(raw / "forecast_live.csv")
    models = [m for m in C.MODELS if m in set(hist.model) and hist[hist.model == m].precip_mm.notna().any()]
    if len(models) < 2:
        raise SystemExit(f"Need >=2 models with data, found {models}. Check data/raw/history_models.csv")
    print("models with data:", models)

    susc = risk.susceptibility(elev)
    susc.to_csv(out / "susceptibility.csv", index=False)

    # ---- 1. backtest of the blend, per locality, no look-ahead
    frames, fac_rows, live_rows = [], [], []
    for loc in locs.locality:
        w = wide_history(hist, obs, loc, models)
        if w["obs"].notna().sum() < 60:
            print("skip (too little history):", loc)
            continue
        b = blend.blend_series(w, models)
        b["locality"] = loc
        frames.append(b.reset_index().rename(columns={"index": "date"}))
        f = blend.live_factors(w, models)
        f["locality"] = loc
        fac_rows.append(f)
        # ---- 2. live forecast with the latest factors
        lv = live[live.locality == loc].pivot(index="date", columns="model", values="precip_mm")
        lv = lv.reindex(columns=models)
        lv.index = pd.to_datetime(lv.index)
        a = blend.apply_live(lv, f, models)
        a["locality"] = loc
        live_rows.append(a.reset_index().rename(columns={"index": "date"}))
    hb = pd.concat(frames, ignore_index=True)
    hb["date"] = pd.to_datetime(hb["date"])
    hb.to_csv(out / "history_blend.csv", index=False)
    pd.concat(fac_rows).to_csv(out / "factors.csv", index=False)

    # ---- validation (monsoon season Jun-Oct; drop 30-day warm-up per locality)
    raw_cols = [f"raw_{m}" for m in models]
    # fair comparison: only days where ALL models and the observation exist, after a 30-day warm-up
    hb["all_ok"] = hb[raw_cols].notna().all(axis=1) & hb["obs"].notna()
    hb["rank"] = hb[hb["all_ok"]].groupby("locality").cumcount()
    ev = hb[(hb["rank"] >= C.BIAS_WINDOW_DAYS) & hb["date"].dt.month.between(6, 10)].copy()
    for m in models:
        first = hb.loc[hb[f"raw_{m}"].notna(), "date"].min()
        print(f"  {m}: forecasts available from {first.date()}")
    if ev.empty:
        raise SystemExit("No Jun-Oct days where all models have data after warm-up. Try fewer models in config.MODELS.")
    cols = raw_cols + ["mean_raw", "mean_corr", "blend"]
    by_year = {str(y): blend.skill(g, cols) for y, g in ev.groupby(ev["date"].dt.year)}
    overall_all = blend.skill(ev, cols)
    hold = ev[ev["date"].dt.year >= C.HOLDOUT_FROM_YEAR]
    if len(hold) < 500:
        print("  (holdout too small; headline metrics use all common monsoon days)")
        hold = ev
    ev_all = ev
    ev = hold
    overall = blend.skill(ev, cols)
    best_single = min((c for c in cols if c.startswith("raw_")), key=lambda c: overall[c]["mae"])
    metrics = {
        "scope": f"daily rainfall, 22 Hyderabad localities, Jun-Oct, unseen holdout seasons (>= {C.HOLDOUT_FROM_YEAR}); settings: ratio_power={C.RATIO_POWER}, weight_power={C.WEIGHT_POWER}",
        "overall_all_years": overall_all,
        "threshold_mm": C.EVENT_THRESHOLD_MM,
        "n_localities": int(ev.locality.nunique()),
        "period": [str(ev.date.min().date()), str(ev.date.max().date())],
        "models": models,
        "best_single_model": best_single,
        "overall": overall,
        "by_year": by_year,
        "mae_gain_vs_best_single_pct": 100 * (1 - overall["blend"]["mae"] / overall[best_single]["mae"]),
        "rmse_gain_vs_best_single_pct": 100 * (1 - overall["blend"]["rmse"] / overall[best_single]["rmse"]),
        "mae_gain_vs_equal_mean_pct": 100 * (1 - overall["blend"]["mae"] / overall["mean_raw"]["mae"]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"MAE gain vs best single: {metrics['mae_gain_vs_best_single_pct']:.1f}%   "
          f"vs equal-weight mean: {metrics['mae_gain_vs_equal_mean_pct']:.1f}%")

    # ---- risk tables
    lv = pd.concat(live_rows, ignore_index=True)
    rf = risk.score_table(lv, susc, locs)
    rf["date"] = pd.to_datetime(rf["date"]).dt.date.astype(str)
    rf.to_csv(out / "risk_forecast.csv", index=False)

    rp_path = raw / "replay_2020.csv"
    if rp_path.exists():
        rp = pd.read_csv(rp_path).rename(columns={"precip_mm": "rain_blend"})
        rr = risk.score_table(rp, susc, locs)
        rr.to_csv(out / "risk_replay_2020.csv", index=False)

    meta = json.loads((raw / "meta.json").read_text()) if (raw / "meta.json").exists() else {}
    meta["processed_models"] = models
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    main(Path(a.raw), Path(a.out))
