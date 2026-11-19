"""JalRaksha - Hyderabad flood early-warning + response dashboard.   streamlit run app.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from jalraksha import config as C
from jalraksha import respond

ROOT = Path(__file__).parent
PROC = Path(os.environ.get("JALRAKSHA_PROCESSED", ROOT / "data" / "processed"))
RAW = Path(os.environ.get("JALRAKSHA_RAW", ROOT / "data" / "raw"))
ROADS = RAW / "hyderabad_roads.graphml"

st.set_page_config(page_title="JalRaksha - Hyderabad Flood Response", page_icon="🌊", layout="wide")

LEVEL_EMOJI = {"Green": "🟢", "Yellow": "🟡", "Orange": "🟠", "Red": "🔴"}
MODEL_LABEL = C.MODEL_LABELS


@st.cache_data
def load(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def need(name: str):
    df = load(str(PROC / name))
    if df is None:
        st.error(f"Missing {PROC / name}. Run `python fetch_data.py` then `python run_pipeline.py` first.")
        st.stop()
    return df


@st.cache_resource
def roads():
    return respond.RoadNet.load(ROADS) if ROADS.exists() else None


risk_fc = need("risk_forecast.csv")
metrics = json.loads((PROC / "metrics.json").read_text())
meta = json.loads((PROC / "meta.json").read_text()) if (PROC / "meta.json").exists() else {}
factors = need("factors.csv")
locs = need("susceptibility.csv")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌊 JalRaksha")
st.sidebar.caption("Flood early-warning and response support for Hyderabad")
rep_df = load(str(PROC / "risk_replay_2020.csv"))
live_has_alerts = bool(risk_fc["level"].isin(["Orange", "Red"]).any())
scen_opts = ["Live 7-day forecast"] + (["Replay: Oct 2020 floods"] if rep_df is not None else [])
scenario = st.sidebar.radio("Scenario", scen_opts, index=(1 if (not live_has_alerts and len(scen_opts) > 1) else 0))
if scenario.startswith("Replay"):
    risk_fc = rep_df.copy()
    risk_fc["date"] = risk_fc["date"].astype(str)
    dates = sorted(risk_fc["date"].unique())
    worst = risk_fc.groupby("date")["score"].sum().idxmax()
    st.sidebar.caption("Replays the Oct 2020 Hyderabad floods using observed rainfall (ERA5-based, which under-reports "
                       "extremes). Use it to see Prepare and Respond in action.")
    day = st.sidebar.select_slider("Day", options=dates, value=worst)
else:
    dates = sorted(risk_fc["date"].unique())
    if not live_has_alerts:
        st.sidebar.info("No Orange/Red zone in this week's forecast. Switch Scenario to the Oct 2020 replay to see the full workflow.")
    day = st.sidebar.select_slider("Forecast day", options=dates, value=dates[min(1, len(dates) - 1)])
st.sidebar.markdown("**Resources on hand**")
res = {
    "Rescue teams": st.sidebar.number_input("Rescue teams", 0, 200, 12),
    "Boats": st.sidebar.number_input("Boats", 0, 200, 8),
    "Dewatering pumps": st.sidebar.number_input("Dewatering pumps", 0, 500, 30),
}
lang = st.sidebar.radio("Alert language", list(respond.LANGS), format_func=lambda k: respond.LANGS[k], horizontal=True)
if meta.get("fixture"):
    st.sidebar.error("⚠️ TEST FIXTURE DATA - not real. Run fetch_data.py.")
st.sidebar.caption(f"Data fetched: {meta.get('fetched_at', 'n/a')}")

rd = risk_fc[risk_fc["date"] == day].copy()
rd["lvl_rank"] = rd["level"].map({k: i for i, k in enumerate(C.LEVEL_ORDER)})
rd = rd.sort_values(["lvl_rank", "score"], ascending=False)


def risk_map(df: pd.DataFrame, paths=None, height=520):
    d = df.copy()
    d["color"] = d["level"].map(lambda l: C.LEVEL_COLORS[l] + [190])
    d["radius"] = 700 + d["score"].clip(0, 200) * 12
    d["tip"] = d.apply(lambda r: f"{r['locality']}: {r['level']} | rain {r['rain_blend']:.0f} mm | "
                                 f"terrain susceptibility {r['susc']:.2f} | confidence {r['confidence']}", axis=1)
    layers = [pdk.Layer("ScatterplotLayer", d, get_position="[lon, lat]", get_fill_color="color",
                        get_radius="radius", pickable=True)]
    for p in paths or []:
        layers.append(pdk.Layer("PathLayer", [{"path": p["coords"]}], get_path="path", get_color=p["color"],
                                width_min_pixels=p.get("w", 4)))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=17.41, longitude=78.47, zoom=10.2),
                             tooltip={"text": "{tip}"}), height=height)


st.title("JalRaksha — Hyderabad flood early-warning & response")
st.caption("Predict → Prepare → Respond. One hazard, one city, built around the decisions of a police commissionerate "
           "and district disaster cell.")
top = rd.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Day shown", day)
c2.metric("Red zones", int((rd.level == "Red").sum()))
c3.metric("Orange zones", int((rd.level == "Orange").sum()))
c4.metric("Highest-risk locality", f"{top['locality']}", f"{top['rain_blend']:.0f} mm blended rain", delta_color="off")

t_pred, t_prep, t_resp, t_val, t_replay, t_about = st.tabs(
    ["1 · Predict", "2 · Prepare", "3 · Respond", "Validation", "Replay: Oct 2020", "About & limits"])

# ------------------------------------------------------------------ PREDICT
with t_pred:
    left, right = st.columns([3, 2])
    with left:
        risk_map(rd)
    with right:
        view = rd[["locality", "level", "rain_blend", "rain_min", "rain_max", "susc", "confidence"]].copy()
        view["level"] = view["level"].map(lambda l: f"{LEVEL_EMOJI[l]} {l}")
        view.columns = ["Locality", "Level", "Blended rain mm", "Min model", "Max model", "Terrain susc.", "Confidence"]
        st.dataframe(view.round(1), hide_index=True, height=520, use_container_width=True)
    st.markdown("**7-day outlook (worst level per day)**")
    piv = risk_fc.pivot_table(index="locality", columns="date", values="score", aggfunc="max")
    fig = px.imshow(piv, color_continuous_scale=["#2ea043", "#f0c800", "#f58214", "#d62728"], aspect="auto",
                    zmin=0, zmax=C.IMD_VERY_HEAVY * 1.2, labels=dict(color="Effective score"))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**How much each model is trusted right now** (MausamMix adaptive weights, last 30 days)")
    fw = factors.copy()
    fw["model"] = fw["model"].map(MODEL_LABEL)
    fig2 = px.imshow(fw.pivot(index="locality", columns="model", values="weight"), color_continuous_scale="Blues",
                     aspect="auto", zmin=0, zmax=1, labels=dict(color="Weight"))
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------ PREPARE
with t_prep:
    st.subheader(f"Pre-positioning plan for {day}")
    alloc = respond.allocate(rd, {k: int(v) for k, v in res.items()})
    alloc = alloc.merge(rd[["locality", "action"]], on="locality")
    alloc = alloc[alloc["level"] != "Green"].sort_values(list(res), ascending=False)
    alloc["level"] = alloc["level"].map(lambda l: f"{LEVEL_EMOJI[l]} {l}")
    if alloc.empty:
        st.success("No Yellow/Orange/Red zones on this day - no pre-positioning needed.")
    else:
        st.dataframe(alloc.rename(columns={"locality": "Locality", "level": "Level", "action": "Recommended action"}),
                     hide_index=True, use_container_width=True)
        st.caption("Resources are split in proportion to warning level × terrain susceptibility; every Red zone gets at "
                   "least one of each resource while stock lasts.")
    st.markdown("**Advance-notice view: when does each zone first turn Orange or Red?**")
    first = (risk_fc[risk_fc.level.isin(["Orange", "Red"])].sort_values("date").groupby("locality")
             .agg(first_alert_date=("date", "first"), worst_level=("level", lambda s: "Red" if "Red" in set(s) else "Orange"))
             .reset_index())
    st.dataframe(first, hide_index=True, use_container_width=True) if len(first) else st.info("No Orange/Red zone in the 7-day window.")

# ------------------------------------------------------------------ RESPOND
with t_resp:
    a, b, c = st.tabs(["📢 Multilingual alerts", "🆘 SOS triage", "🛣️ Flood-aware routes"])
    with a:
        pick = st.selectbox("Locality", rd["locality"].tolist())
        row = rd[rd.locality == pick].iloc[0]
        txt = respond.draft_alert(row["level"], pick, day, row["rain_blend"], lang)
        if txt:
            st.text_area("Draft public alert (edit before sending)", txt, height=130)
            with st.expander("Same alert in all three languages"):
                for k, nm in respond.LANGS.items():
                    st.markdown(f"**{nm}**")
                    st.write(respond.draft_alert(row["level"], pick, day, row["rain_blend"], k))
        else:
            st.info(f"{pick} is Green on {day}: no alert needed.")
        st.caption("Templates are drafts: have a native speaker and the PIO verify wording before real use.")
    with b:
        st.markdown("Citizen reports (WhatsApp/112/helpline) are scored on danger keywords, people at risk, water depth "
                    "and the zone's flood level. **The sample file is simulated** for demo only.")
        up = st.file_uploader("Upload reports CSV (report_id, locality, text, people, depth_cm)", type="csv")
        rep = pd.read_csv(up) if up else pd.read_csv(ROOT / "data" / "demo_sos_reports_SIMULATED.csv")
        tri = respond.triage(rep, rd)
        st.dataframe(tri[["priority", "score", "locality", "text", "people", "depth_cm", "why"]],
                     hide_index=True, use_container_width=True)
    with c:
        net = roads()
        if net is None:
            st.info("Road graph not downloaded. Run `python fetch_data.py --roads` to enable routing.")
        else:
            names = {"Command centre (approx.)": C.COMMAND_CENTRE[1:]}
            names.update({r.locality: (r.lat, r.lon) for r in rd.itertuples()})
            o = st.selectbox("From", list(names), index=0)
            d = st.selectbox("To (incident / relief point)", [n for n in names if n != o], index=2)
            zones = rd[["locality", "level", "lat", "lon"]]
            rt = net.route(names[o], names[d], zones)
            paths = [dict(coords=rt["direct"]["coords"], color=[120, 120, 120], w=3),
                     dict(coords=rt["safe"]["coords"], color=[20, 90, 220], w=6)]
            risk_map(rd, paths, height=480)
            m1, m2 = st.columns(2)
            m1.metric("Direct route", f"{rt['direct']['km']:.1f} km", f"{rt['direct']['flooded_km']:.1f} km through Orange/Red zones",
                      delta_color="inverse")
            m2.metric("Flood-aware route (blue)", f"{rt['safe']['km']:.1f} km", f"{rt['safe']['flooded_km']:.1f} km through Orange/Red zones",
                      delta_color="inverse")
            st.caption("Roads within 1.5 km of an Orange/Red locality are penalised (×4 / ×25). Major roads only.")

# ------------------------------------------------------------------ VALIDATION
with t_val:
    ov = metrics["overall"]
    st.subheader("Does the blend actually beat the individual models?")
    st.caption(f"{metrics['scope']} · {metrics['period'][0]} to {metrics['period'][1]} · "
               f"{metrics['n_localities']} localities · bias/weights learned only from past data (no look-ahead).")
    rows = []
    names = {**{f"raw_{m}": f"{MODEL_LABEL[m]} (raw)" for m in metrics["models"]},
             "mean_raw": "Equal-weight mean (raw)", "mean_corr": "Equal-weight mean (bias-corrected)",
             "blend": "JalRaksha blend (MausamMix)"}
    for k, v in ov.items():
        rows.append({"Forecast": names[k], "MAE mm/day": v["mae"], "RMSE mm/day": v["rmse"], "Bias mm/day": v["bias"],
                     f"Detects ≥{metrics['threshold_mm']:.0f} mm (POD)": v["pod"], "False alarm ratio": v["far"], "CSI": v["csi"]})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.round(3), hide_index=True, use_container_width=True)
    g1, g2 = metrics["mae_gain_vs_best_single_pct"], metrics["mae_gain_vs_equal_mean_pct"]
    best = names[metrics["best_single_model"]]
    if g1 > 3:
        st.success(f"Blend MAE is **{g1:.1f}% lower** than the best single model ({best}) and {g2:.1f}% lower than a plain average.")
    elif g1 >= -3:
        st.info(f"The blend **matches the best single model** ({best}) within {abs(g1):.1f}% MAE, and is **{g2:.1f}% better than a plain "
                "average** of the models. Its value is that it finds the best model by itself, locality by locality.")
    else:
        st.warning(f"On this sample the blend did **not** beat the best single model ({best}): MAE change {g1:.1f}%. We report this as-is.")
    fig = px.bar(tbl.melt(id_vars="Forecast", value_vars=["MAE mm/day", "RMSE mm/day"]), x="Forecast", y="value",
                 color="variable", barmode="group")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="mm/day")
    st.plotly_chart(fig, use_container_width=True)
    yr = pd.DataFrame([{"Year": y, "Forecast": names[k], "MAE": v["mae"]} for y, d in metrics["by_year"].items() for k, v in d.items()])
    st.plotly_chart(px.bar(yr, x="Year", y="MAE", color="Forecast", barmode="group").update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0)),
                    use_container_width=True)
    st.caption("Truth = Open-Meteo Historical Weather (ERA5-based reanalysis). ERA5 rainfall comes from ECMWF's own modelling "
               "system, so it probably flatters the ECMWF forecast; it is also smoother than gauge data and underestimates extremes. "
               "Verification against IMD / GHMC rain-gauge data is the key next step. Headline numbers are from unseen holdout "
               "seasons; settings were tuned on earlier seasons only.")

# ------------------------------------------------------------------ REPLAY
with t_replay:
    rep20 = load(str(PROC / "risk_replay_2020.csv"))
    st.subheader("Replay: what would JalRaksha have shown during the October 2020 Hyderabad floods?")
    if rep20 is None:
        st.info("Replay data not available.")
    else:
        dts = sorted(rep20["date"].unique())
        dsel = st.select_slider("Day", options=dts, value=max(dts, key=lambda x: rep20[rep20.date == x]["rain_blend"].sum()))
        r2 = rep20[rep20.date == dsel]
        risk_map(r2, height=450)
        daily = rep20.groupby("date").agg(max_rain=("rain_blend", "max"), red=("level", lambda s: (s == "Red").sum()),
                                           orange=("level", lambda s: (s == "Orange").sum())).reset_index()
        st.plotly_chart(px.bar(daily, x="date", y="max_rain", hover_data=["red", "orange"], labels={"max_rain": "Max locality rain, mm/day"})
                        .update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
        st.caption("Observation-driven replay (ERA5-based rainfall, not a historical forecast). Reanalysis typically "
                   "under-represents extreme urban rainfall, so levels shown are a conservative floor.")

# ------------------------------------------------------------------ ABOUT
with t_about:
    st.markdown((ROOT / "docs" / "ABOUT.md").read_text())
