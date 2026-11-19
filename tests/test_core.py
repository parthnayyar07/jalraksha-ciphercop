import numpy as np
import pandas as pd
import networkx as nx

from jalraksha import blend, risk, respond
from jalraksha import config as C

MODELS = ["a", "b", "c"]


def _df(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=n)
    obs = np.where(rng.random(n) < .4, rng.gamma(1, 15, n), 0)
    d = pd.DataFrame({m: np.clip(obs * f + rng.normal(0, 2, n), 0, None) for m, f in zip(MODELS, (1.2, .7, 1.0))}, index=idx)
    d["obs"] = obs
    return d


def test_no_lookahead():
    d = _df()
    t = int(np.where((d[MODELS] > 5).all(axis=1).values & (np.arange(len(d)) > 60))[0][0])
    base = blend.blend_series(d, MODELS)
    d2 = d.copy()
    # change observations at t-1, t and after: forecast for t may only use obs <= t-2
    d2.iloc[t - 1:, d2.columns.get_loc("obs")] += 500
    chg = blend.blend_series(d2, MODELS)
    assert np.isclose(base["blend"].iloc[t], chg["blend"].iloc[t])
    # but it must react to changes at t-2 (sanity: the test is sensitive)
    d3 = d.copy()
    d3.iloc[t - 2, d3.columns.get_loc("obs")] += 500
    assert not np.isclose(base["blend"].iloc[t], blend.blend_series(d3, MODELS)["blend"].iloc[t])


def test_blend_nonnegative_and_weights_sum_to_one():
    b = blend.blend_series(_df(), MODELS)
    assert (b["blend"].dropna() >= 0).all()
    w = b[[f"w_{m}" for m in MODELS]].sum(axis=1)
    assert np.allclose(w, 1.0)


def test_bias_correction_helps_biased_model():
    d = _df(300)
    b = blend.blend_series(d, MODELS, ratio_power=1.0).iloc[40:]   # mechanism works when enabled (default is off for rainfall)
    raw = (b["raw_a"] - b["obs"]).abs().mean()
    cor = (b["corr_a"] - b["obs"]).abs().mean()
    assert cor < raw


def test_live_factors_and_apply():
    d = _df()
    f = blend.live_factors(d, MODELS)
    assert abs(f.weight.sum() - 1) < 1e-9
    fc = pd.DataFrame({m: [10., 80.] for m in MODELS}, index=pd.date_range("2024-01-01", periods=2))
    out = blend.apply_live(fc, f, MODELS)
    assert (out.rain_min <= out.rain_blend + 1e-9).all() and (out.rain_blend <= out.rain_max + 1e-9).all()


def test_levels_follow_imd():
    assert risk.level_from_score(5) == "Green"
    assert risk.level_from_score(C.IMD_MODERATE) == "Yellow"
    assert risk.level_from_score(C.IMD_HEAVY) == "Orange"
    assert risk.level_from_score(C.IMD_VERY_HEAVY) == "Red"
    assert risk.effective_score(100, 1.0) > risk.effective_score(100, 0.0)


def test_susceptibility_range_and_order():
    e = pd.DataFrame({"locality": list("abcd"), "elev_m": [480, 500, 520, 540], "ring_mean_m": [510, 505, 520, 530]})
    s = risk.susceptibility(e).set_index("locality")["susc"]
    assert s.between(0, 1).all() and s["a"] > s["d"]


def test_allocation_totals_and_red_priority():
    r = pd.DataFrame({"locality": list("abcd"), "level": ["Red", "Orange", "Yellow", "Green"], "susc": [.9, .5, .5, .5]})
    a = respond.allocate(r, {"boats": 5, "pumps": 3})
    assert a["boats"].sum() == 5 and a["pumps"].sum() == 3
    assert a.loc[a.locality == "a", "boats"].iloc[0] >= 1
    assert a.loc[a.locality == "d", "boats"].iloc[0] == 0


def test_triage_orders_by_danger():
    risk_day = pd.DataFrame({"locality": ["X", "Y"], "level": ["Red", "Green"]})
    rep = pd.DataFrame([
        {"report_id": 1, "locality": "Y", "text": "water in the street", "people": 0, "depth_cm": 10},
        {"report_id": 2, "locality": "X", "text": "elderly woman trapped, oxygen needed, live wire nearby", "people": 3, "depth_cm": 90},
    ])
    t = respond.triage(rep, risk_day)
    assert t.iloc[0]["report_id"] == 2 and t.iloc[0]["priority"].startswith("P1")
    assert t.iloc[1]["priority"].startswith("P3")


def test_alerts_all_languages():
    for lvl in ("Yellow", "Orange", "Red"):
        for lg in respond.LANGS:
            s = respond.draft_alert(lvl, "Malakpet", "2024-11-08", 87.4, lg)
            assert "Malakpet" in s and "87" in s and "112" in s or lvl == "Yellow"
    assert respond.draft_alert("Green", "x", "d", 1, "en") == ""


def test_routing_avoids_flooded_zone():
    # 5x5 grid roads; flood zone sits on the straight line between origin and destination
    G = nx.MultiDiGraph()
    lat0, lon0, step = 17.40, 78.40, 0.01
    for i in range(5):
        for j in range(5):
            G.add_node(f"{i}_{j}", y=lat0 + i * step, x=lon0 + j * step)
    for i in range(5):
        for j in range(5):
            for di, dj in ((1, 0), (0, 1)):
                a, b = f"{i}_{j}", f"{i+di}_{j+dj}"
                if f"{i+di}_{j+dj}" in G.nodes:
                    G.add_edge(a, b, length=1100.0)
                    G.add_edge(b, a, length=1100.0)
    net = respond.RoadNet(G)
    zones = pd.DataFrame({"locality": ["flood"], "level": ["Red"], "lat": [lat0 + 2 * step], "lon": [lon0 + 2 * step]})
    r = net.route((lat0, lon0), (lat0 + 4 * step, lon0 + 4 * step), zones, radius_km=0.8)
    assert r["safe"]["flooded_km"] < r["direct"]["flooded_km"] or r["direct"]["flooded_km"] == 0
    assert r["safe"]["km"] >= r["direct"]["km"] - 1e-9
    assert r["safe"]["flooded_km"] == 0


def test_blend_survives_model_starting_late():
    d = _df(200)
    d.loc[d.index[:90], "c"] = np.nan          # model c has no history for the first 90 days
    b = blend.blend_series(d, MODELS)
    assert b["blend"].notna().all()
    assert np.allclose(b.loc[b.index[:90], "w_c"], 0.0)
    assert np.allclose(b[[f"w_{m}" for m in MODELS]].sum(axis=1), 1.0)
