"""Terrain susceptibility + rainfall-to-warning-level logic (transparent heuristic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def susceptibility(elev: pd.DataFrame) -> pd.DataFrame:
    """elev: locality, elev_m, ring_mean_m. Returns susc in [0,1] (1 = most low-lying).

    Low-lying relative to surroundings (water collects) and low absolute elevation both raise
    susceptibility. Percentile ranks across localities keep it robust to the units of relief.
    """
    d = elev.copy()
    d["rel_elev_m"] = d["elev_m"] - d["ring_mean_m"]
    rank_rel = d["rel_elev_m"].rank(ascending=False, pct=True)   # most negative rel_elev -> ~1
    rank_abs = d["elev_m"].rank(ascending=False, pct=True)       # lowest absolute elev -> ~1
    d["susc"] = (0.6 * rank_rel + 0.4 * rank_abs).round(3)
    lo, hi = d["susc"].min(), d["susc"].max()
    d["susc"] = ((d["susc"] - lo) / (hi - lo)).round(3) if hi > lo else 0.5
    return d[["locality", "elev_m", "ring_mean_m", "rel_elev_m", "susc"]]


def effective_score(rain_mm: float, susc: float) -> float:
    """Rainfall amplified (up to +25%) or damped (down to -25%) by terrain susceptibility."""
    return float(rain_mm) * (0.75 + 0.5 * float(susc))


def level_from_score(score: float) -> str:
    if score >= C.IMD_VERY_HEAVY:
        return "Red"
    if score >= C.IMD_HEAVY:
        return "Orange"
    if score >= C.IMD_MODERATE:
        return "Yellow"
    return "Green"


def confidence(rain_blend: float, rmin: float, rmax: float) -> str:
    if rain_blend < 5 or np.isnan(rain_blend):
        return "High"
    spread = (rmax - rmin) / max(rain_blend, 1.0)
    return "High" if spread < 0.4 else "Medium" if spread < 0.8 else "Low"


ACTIONS = {
    "Green": "No flood action. Routine monitoring.",
    "Yellow": "Be aware. Check drains/culverts, keep control room on watch, brief traffic units.",
    "Orange": "Be prepared. Pre-position pumps, boats and rescue teams; plan diversions; issue public advisory.",
    "Red": "Take action. Activate evacuation of low-lying pockets, close flooded underpasses, "
           "deploy rescue teams, issue multilingual alerts.",
}


def score_table(rain: pd.DataFrame, susc: pd.DataFrame, locs: pd.DataFrame) -> pd.DataFrame:
    """rain: locality,date,rain_blend[,rain_min,rain_max]. Returns full risk rows."""
    df = rain.merge(susc[["locality", "susc", "elev_m", "rel_elev_m"]], on="locality", how="left")
    df = df.merge(locs[["locality", "lat", "lon"]], on="locality", how="left")
    if "rain_min" not in df:
        df["rain_min"] = df["rain_blend"]
        df["rain_max"] = df["rain_blend"]
    df["susc"] = df["susc"].fillna(0.5)
    df["score"] = [effective_score(r, s) for r, s in zip(df["rain_blend"], df["susc"])]
    df["level"] = df["score"].map(level_from_score)
    df["confidence"] = [confidence(b, lo, hi) for b, lo, hi in zip(df["rain_blend"], df["rain_min"], df["rain_max"])]
    df["action"] = df["level"].map(ACTIONS)
    return df
