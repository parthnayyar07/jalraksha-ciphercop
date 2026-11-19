"""MausamMix-style adaptive rainfall blending.

Per locality and day:
  1. (optional) bias correction: multiply each model by a trailing 30-day ratio sum(obs)/sum(forecast)^RATIO_POWER.
     Experiments (experiments.py) showed this HURTS rainfall error, so config.RATIO_POWER = 0 (off).
  2. Adaptive weights: weight each model by (1 / (recent MAE + 0.5)) ** WEIGHT_POWER (default 4), so the model
     that has been best nearby dominates, and models that are missing on a day get weight 0.
  3. No look-ahead: a forecast for day t only uses observations up to day t - LAG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _trailing(series: pd.Series, how: str, window: int, lag: int, min_periods: int) -> pd.Series:
    r = series.shift(lag).rolling(window, min_periods=min_periods)
    return r.sum() if how == "sum" else r.mean()


def blend_series(
    df: pd.DataFrame,
    models: list[str],
    window: int = C.BIAS_WINDOW_DAYS,
    lag: int = C.BIAS_LAG_DAYS,
    min_pairs: int = 10,
    clip: tuple[float, float] = C.BIAS_RATIO_CLIP,
    ratio_power: float | None = None,
    weight_power: float | None = None,
) -> pd.DataFrame:
    """df: DatetimeIndex, columns = models + 'obs'. Returns raw/corr/weights/blend per day."""
    rp = C.RATIO_POWER if ratio_power is None else ratio_power
    wp = C.WEIGHT_POWER if weight_power is None else weight_power
    out = pd.DataFrame(index=df.index)
    obs = df["obs"]
    maes = {}
    for m in models:
        fc = df[m]
        ok = fc.notna() & obs.notna()
        s_fc = _trailing(fc.where(ok), "sum", window, lag, min_pairs)
        s_ob = _trailing(obs.where(ok), "sum", window, lag, min_pairs)
        ratio = ((s_ob + 1.0) / (s_fc + 1.0)).clip(*clip) ** rp
        ratio = ratio.where(s_fc >= 5.0, 1.0).fillna(1.0)
        corr = (fc * ratio).clip(lower=0)
        err = (corr - obs).abs().where(ok)
        maes[m] = _trailing(err, "mean", window, lag, min_pairs)
        out[f"raw_{m}"] = fc
        out[f"corr_{m}"] = corr
        out[f"ratio_{m}"] = ratio
    corr_cols = [f"corr_{m}" for m in models]
    corr_mat = out[corr_cols]
    avail = corr_mat.notna().values
    inv = pd.DataFrame({m: (1.0 / (maes[m] + 0.5)) ** wp for m in models})
    inv = inv.where(inv.notna(), 1.0)                       # warm-up: equal trust
    inv = inv.where(pd.DataFrame(avail, index=inv.index, columns=models), 0.0)  # missing model -> weight 0
    tot = inv.sum(axis=1)
    w = inv.div(tot.where(tot > 0), axis=0).fillna(0.0)
    for m in models:
        out[f"w_{m}"] = w[m]
    out["blend"] = (corr_mat.fillna(0.0).values * w[models].values).sum(axis=1)
    out.loc[~avail.any(axis=1), "blend"] = np.nan
    out["mean_raw"] = out[[f"raw_{m}" for m in models]].mean(axis=1)
    out["mean_corr"] = corr_mat.mean(axis=1)
    out["obs"] = obs
    return out


def live_factors(df: pd.DataFrame, models: list[str], window: int = C.BIAS_WINDOW_DAYS,
                 clip: tuple[float, float] = C.BIAS_RATIO_CLIP) -> pd.DataFrame:
    """Ratios + weights from the most recent `window` paired days, to apply to live forecasts."""
    d = df.dropna(subset=["obs"]).tail(window)
    rows, inv = [], {}
    for m in models:
        ok = d[m].notna()
        s_fc, s_ob = d.loc[ok, m].sum(), d.loc[ok, "obs"].sum()
        ratio = float(np.clip((s_ob + 1.0) / (s_fc + 1.0), *clip)) ** C.RATIO_POWER if s_fc >= 5.0 else 1.0
        mae = float((d.loc[ok, m] * ratio - d.loc[ok, "obs"]).abs().mean()) if ok.any() else np.nan
        inv[m] = (1.0 / (mae + 0.5)) ** C.WEIGHT_POWER if not np.isnan(mae) else np.nan
        rows.append({"model": m, "ratio": ratio, "mae_recent": mae})
    tot = np.nansum(list(inv.values()))
    for r in rows:
        r["weight"] = (inv[r["model"]] / tot) if tot > 0 and not np.isnan(inv[r["model"]]) else 1 / len(models)
    return pd.DataFrame(rows)


def apply_live(fc_wide: pd.DataFrame, factors: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """fc_wide: index=date, columns=models (raw forecasts). Returns blend, min/max spread."""
    f = factors.set_index("model")
    corr = pd.DataFrame({m: (fc_wide[m] * f.loc[m, "ratio"]).clip(lower=0) for m in models})
    w = np.array([f.loc[m, "weight"] for m in models])
    vals = corr[models].values
    mask = ~np.isnan(vals)
    wsum = (mask * w).sum(axis=1)
    blend = np.where(wsum > 0, np.nansum(vals * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    return pd.DataFrame({
        "rain_blend": blend,
        "rain_min": corr.min(axis=1),
        "rain_max": corr.max(axis=1),
    }, index=fc_wide.index)


def skill(df: pd.DataFrame, cols: list[str], thr: float = C.EVENT_THRESHOLD_MM) -> dict:
    """Pooled error + event-detection skill for each column vs df['obs']."""
    res = {}
    for c in cols:
        d = df[[c, "obs"]].dropna()
        if d.empty:
            continue
        e = d[c] - d["obs"]
        hit = int(((d[c] >= thr) & (d["obs"] >= thr)).sum())
        miss = int(((d[c] < thr) & (d["obs"] >= thr)).sum())
        fa = int(((d[c] >= thr) & (d["obs"] < thr)).sum())
        res[c] = {
            "n": int(len(d)),
            "mae": float(e.abs().mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()),
            "pod": hit / (hit + miss) if hit + miss else None,
            "far": fa / (hit + fa) if hit + fa else None,
            "csi": hit / (hit + miss + fa) if hit + miss + fa else None,
            "events": hit + miss,
        }
    return res
