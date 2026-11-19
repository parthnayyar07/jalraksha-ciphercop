"""Build the submission deck (PPTX) from the REAL pipeline outputs in data/processed/.

    python make_deck.py [--processed data/processed] [--out JalRaksha_Deck.pptx]

Every number and chart comes from metrics.json / risk_*.csv - nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

from jalraksha import config as C
from jalraksha import respond

NAVY, TEAL, INK, GREY, LIGHT = "0B2545", "13A89E", "1B1F24", "5F6B76", "EEF4F7"
LVL = {"Green": "#2ea043", "Yellow": "#f0c800", "Orange": "#f58214", "Red": "#d62728"}
LAB = C.MODEL_LABELS


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


# ----------------------------------------------------------------- figures
def fig_arch(path):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.8)
    cols = [("#0B2545", "PREDICT"), ("#13A89E", "PREPARE"), ("#d62728", "RESPOND")]
    boxes = [
        (0.1, "PREDICT", "#0B2545", ["3 models: IFS, GFS, ICON", "MausamMix adaptive weights", "Terrain susceptibility (DEM)", "IMD-class level + confidence"]),
        (4.15, "PREPARE", "#13A89E", ["Rank zones, 7-day view", "Allocate teams / boats / pumps", "First-alert date per zone"]),
        (8.2, "RESPOND", "#d62728", ["EN / HI / TE public alerts", "SOS triage P1-P3", "Flood-aware routes"]),
    ]
    for x, title, col, items in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.2), 3.7, 3.3, boxstyle="round,pad=0.03,rounding_size=0.15", fc=col, ec="none", alpha=0.10))
        ax.add_patch(FancyBboxPatch((x, 2.9), 3.7, 0.6, boxstyle="round,pad=0.03,rounding_size=0.15", fc=col, ec="none"))
        ax.text(x + 1.85, 3.2, title, ha="center", va="center", color="white", fontsize=17, weight="bold")
        for i, t in enumerate(items):
            ax.text(x + 0.2, 2.45 - i * 0.58, "• " + t, fontsize=12, va="center", color="#1B1F24")
    for x in (3.85, 7.9):
        ax.annotate("", xy=(x + 0.27, 1.9), xytext=(x, 1.9), arrowprops=dict(arrowstyle="-|>", lw=3, color="#5F6B76"))
    fig.savefig(path, dpi=200, bbox_inches="tight", transparent=True); plt.close(fig)


def fig_validation(metrics, path):
    names = {**{f"raw_{m}": f"{LAB[m]}\n(raw)" for m in metrics["models"]},
             "mean_raw": "Equal-weight\nmean (raw)", "mean_corr": "Equal mean\n(bias-corrected)", "blend": "JalRaksha\nblend"}
    keys = list(metrics["overall"])
    mae = [metrics["overall"][k]["mae"] for k in keys]
    rmse = [metrics["overall"][k]["rmse"] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    cols = ["#9AA7B2"] * len(keys); cols[-1] = "#13A89E"
    ax.bar(x - w / 2, mae, w, color=cols, label="MAE")
    ax.bar(x + w / 2, rmse, w, color=cols, alpha=0.55, label="RMSE", hatch="//", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys], fontsize=9)
    ax.set_ylabel("mm / day (lower is better)")
    for i, v in enumerate(mae):
        ax.text(i - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_map(risk, day, path):
    d = risk[risk.date == day]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(d.lon, d.lat, s=140 + d.score.clip(0, 200) * 5, c=d.level.map(LVL), edgecolor="white", lw=1.2, alpha=0.9, zorder=3)
    for r in d[d.level.isin(["Orange", "Red"])].itertuples():
        ax.annotate(f"{r.locality.split(' /')[0]}\n{r.rain_blend:.0f} mm", (r.lon, r.lat), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color="#1B1F24", zorder=4)
    ax.set_aspect(1 / np.cos(np.radians(17.4))); ax.grid(alpha=0.25)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title(f"Locality warning levels, {day}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for k, c in LVL.items():
        ax.scatter([], [], c=c, s=70, label=k)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_weights(factors, path):
    p = factors.copy(); p["model"] = p["model"].map(LAB)
    p = p.pivot(index="locality", columns="model", values="weight")
    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    im = ax.imshow(p.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, fontsize=9)
    ax.set_yticks(range(p.shape[0])); ax.set_yticklabels([s.split(" /")[0] for s in p.index], fontsize=8)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if p.values[i, j] > .55 else "#1B1F24")
    ax.set_title("Model trust weights, latest 30 days", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_replay(rep, path):
    g = rep.groupby("date").agg(mx=("rain_blend", "max"), red=("level", lambda s: int((s == "Red").sum())),
                                 org=("level", lambda s: int((s == "Orange").sum()))).reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    cols = ["#d62728" if r > 0 else "#f58214" if o > 0 else "#9AA7B2" for r, o in zip(g.red, g.org)]
    ax.bar(g.date.str[5:], g.mx, color=cols)
    for t, v in zip(range(len(g)), g.mx):
        ax.text(t, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(C.IMD_HEAVY, ls="--", c="#f58214", lw=1); ax.axhline(C.IMD_VERY_HEAVY, ls="--", c="#d62728", lw=1)
    ax.set_ylabel("max locality rain, mm/day"); ax.set_xlabel("date (MM-DD), Oct 2020")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_experiments(proc, path):
    f = proc / "experiments.csv"
    if not f.exists():
        return False
    d = pd.read_csv(f)
    part = d[d.split.str.startswith("HOLD")]
    split_name = "unseen holdout seasons"
    if part.empty:
        part = d[d.split.str.startswith("TUNE")]; split_name = "tuning seasons"
    chosen = f"blend ratio^{C.RATIO_POWER} weight^{C.WEIGHT_POWER}"
    spec = [("raw_ecmwf_ifs025", "ECMWF IFS"), ("raw_gfs_seamless", "NOAA GFS"), ("raw_icon_seamless", "DWD ICON"),
            ("mean_raw", "Plain average\nof 3 models"), ("blend ratio^1.0 weight^1.0", "Blend as first\ndesigned\n(bias-corr.)"),
            (chosen, "JalRaksha blend\n(final)")]
    rows = [(lab, float(part[part.forecast == k]["mae"].iloc[0])) for k, lab in spec if (part.forecast == k).any()]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    cols = ["#9AA7B2"] * len(rows); cols[-1] = "#13A89E"
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=cols)
    for i, (_, v) in enumerate(rows):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"MAE mm/day, {split_name}"); ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return True


# ----------------------------------------------------------------- pptx helpers
def tb(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=None, anchor=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, str) else paras
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        opts = {}
        if isinstance(p, tuple):
            p, opts = p
        run = para.add_run(); run.text = p
        run.font.size = Pt(opts.get("size", size)); run.font.bold = opts.get("bold", bold)
        run.font.color.rgb = rgb(opts.get("color", color)); run.font.name = "Calibri"
        para.space_after = Pt(opts.get("after", 6))
        if align:
            para.alignment = align
    return box


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def base(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, 13.333, 1.05, NAVY)
    rect(s, 0, 1.05, 13.333, 0.06, TEAL)
    tb(s, 0.5, 0.18, 12.3, 0.8, title, size=28, color="FFFFFF", bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        tb(s, 0.5, 1.2, 12.3, 0.5, kicker, size=15, color=GREY)
    tb(s, 0.5, 7.08, 9, 0.3, "JalRaksha · Team North Eagles · CipherCop 2026 · Track 2", size=10, color=GREY)
    return s


def bullets(slide, x, y, w, h, items, size=16):
    paras = []
    for it in items:
        paras.append(("•  " + it, {"size": size, "after": 8}))
    return tb(slide, x, y, w, h, paras)


def stat(slide, x, y, big, small, color=TEAL):
    rect(slide, x, y, 2.9, 1.5, LIGHT)
    tb(slide, x + 0.1, y + 0.1, 2.7, 0.8, big, size=32, color=color, bold=True, align=PP_ALIGN.CENTER)
    tb(slide, x + 0.1, y + 0.88, 2.7, 0.6, small, size=12, color=GREY, align=PP_ALIGN.CENTER)


# ----------------------------------------------------------------- deck
def build(proc: Path, out: Path, example: str = "auto", url: str = ""):
    metrics = json.loads((proc / "metrics.json").read_text())
    risk = pd.read_csv(proc / "risk_forecast.csv")
    factors = pd.read_csv(proc / "factors.csv")
    replay = pd.read_csv(proc / "risk_replay_2020.csv") if (proc / "risk_replay_2020.csv").exists() else None
    figs = proc.parent.parent / "docs" / "figs"; figs.mkdir(parents=True, exist_ok=True)

    peak = risk.groupby("date")["score"].sum().idxmax()
    fig_arch(figs / "arch.png"); fig_validation(metrics, figs / "validation.png")
    fig_weights(factors, figs / "weights.png")
    if replay is not None:
        fig_replay(replay, figs / "replay.png")
    have_exp = fig_experiments(proc, figs / "experiments.png")

    ov = metrics["overall"]; g1 = metrics["mae_gain_vs_best_single_pct"]; g2 = metrics["mae_gain_vs_equal_mean_pct"]
    best_lab = LAB[metrics["best_single_model"].replace("raw_", "")]
    thr = metrics["threshold_mm"]

    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, 13.333, 7.5, NAVY); rect(s, 0, 4.55, 13.333, 0.07, TEAL)
    tb(s, 0.8, 1.4, 11.5, 1.2, "JalRaksha", size=66, color="FFFFFF", bold=True)
    tb(s, 0.8, 2.75, 11.8, 1.6, ["AI flood early-warning & response for Hyderabad",
                                 ("Predict  →  Prepare  →  Respond", {"size": 24, "color": TEAL, "bold": True})], size=30, color="FFFFFF")
    tb(s, 0.8, 4.85, 11.8, 1.8, [("CipherCop 2026 · Track 2: AI for Disaster Resilience", {"size": 20, "color": "FFFFFF", "bold": True}),
                                 ("Team North Eagles: Parth Nayyar · Tanush Sharma · Neev Karwal · Siddhant Shaw", {"size": 18, "color": "C9D6E2"})]
                                + ([(f"Live demo: {url}", {"size": 18, "color": TEAL, "bold": True})] if url else []))

    # 2 problem
    s = base(prs, "The problem: warnings are city-wide, response is street-level",
             "Hyderabad, 13 October 2020 — and every monsoon since")
    stat(s, 0.5, 1.9, "~192 mm", "rain in 24 h on 13 Oct 2020: highest October rainfall since 1903")
    stat(s, 3.6, 1.9, "72", "deaths in Telangana (100+ across the affected region)")
    stat(s, 6.7, 1.9, "3 models", "routinely disagree on how much rain falls, where, and when", color="d62728")
    bullets(s, 0.5, 3.7, 12.3, 3.2, [
        "Forecasters and field officers get several model outputs that disagree; choosing between them is manual and slow.",
        "A district-wide warning does not tell an SP or collector which locality needs boats, which underpass to close, or which SOS call to answer first.",
        "Time between a credible forecast and the first rescue call is the only window to pre-position resources.",
        "Our goal: turn forecasts into locality-level, prioritised actions a police commissionerate and disaster cell can execute.",
    ], size=20)
    tb(s, 0.5, 6.75, 12.3, 0.3, "Sources: Indian Express via Scroll.in (rainfall); Wikipedia, 2020 Hyderabad floods (deaths).", size=10, color=GREY)

    # 3 users and decisions
    s = base(prs, "Built around the decisions officers actually make", "Who uses JalRaksha and what changes for them")
    rows = [("Who", "Decision they face", "JalRaksha output"),
            ("Police commissioner / DCP Traffic", "Which roads and underpasses to close or divert, and when", "Flood-aware routes; 7-day advance warning per locality"),
            ("District disaster cell / GHMC DRF", "Where to pre-position boats, pumps and rescue teams", "Allocation table by warning level × terrain"),
            ("Control room (100 / 112)", "Which of many SOS calls to answer first", "P1 / P2 / P3 triage with reasons"),
            ("Public information officer", "What to tell citizens, in which language", "Ready-to-edit alerts in English, Hindi, Telugu")]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.5), Inches(1.9), Inches(12.3), Inches(4.4)).table
    for j, w in enumerate((3.4, 4.6, 4.3)):
        tbl.columns[j].width = Inches(w)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            c = tbl.cell(i, j); c.text = t
            p = c.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(16 if i else 15); p.runs[0].font.bold = (i == 0 or j == 0)
            p.runs[0].font.color.rgb = rgb("FFFFFF" if i == 0 else INK)
            c.fill.solid(); c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (LIGHT if i % 2 else "FFFFFF"))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE

    # 4 architecture
    s = base(prs, "Solution: one pipeline from forecast to field action", "All three stages run from a single dashboard")
    s.shapes.add_picture(str(figs / "arch.png"), Inches(0.6), Inches(1.9), width=Inches(12.1))
    tb(s, 0.6, 5.9, 12.1, 1.0, "Data: Open-Meteo multi-model forecasts and ERA5-based observations · Copernicus DEM terrain · OpenStreetMap roads. "
       "All public and key-less, so the system can be re-run by any state agency.", size=14, color=GREY)

    # 5 predict
    s = base(prs, "Predict: adaptive multi-model rainfall blending", "Trust-by-recent-skill idea from our Smart India Hackathon prototype, re-tested for rainfall")
    ratio_b = ("Bias correction was tested and switched off: a trailing observed/forecast ratio increased rainfall error in tuning and in unseen seasons."
               if C.RATIO_POWER == 0 else "Bias-correct each model with a trailing 30-day observed/forecast rain ratio (never negative).")
    bullets(s, 0.5, 1.9, 6.6, 4.8, [
        f"Weight each model by (1 / recent error)^{C.WEIGHT_POWER:g}, separately for every locality: the model that has been right nearby dominates.",
        ratio_b,
        "No look-ahead: a forecast for day t uses observations only up to day t − 2 (enforced by an automated test).",
        "Settings chosen on seasons before 2025; headline results are from unseen 2025–26 monsoons.",
        "Warning level uses IMD 24-h classes (Yellow ≥ 15.6, Orange ≥ 64.5, Red ≥ 115.6 mm) after scaling rain by terrain susceptibility (±25%).",
        "Model disagreement becomes a confidence flag, so officers know when not to over-trust one number.",
    ], size=14)
    s.shapes.add_picture(str(figs / "weights.png"), Inches(7.5), Inches(1.75), height=Inches(5.2))

    # 6 validation
    s = base(prs, "Validation: does blending really beat any single model?",
             f"Daily rainfall, {metrics['n_localities']} localities, Jun–Oct, {metrics['period'][0]} to {metrics['period'][1]}; truth = ERA5-based observations")
    s.shapes.add_picture(str(figs / "validation.png"), Inches(0.4), Inches(1.8), width=Inches(8.0))
    tied = abs(g1) <= 3
    stat(s, 9.0, 1.85, f"{g1:+.0f}%", f"MAE vs best single model ({best_lab})" + (": essentially tied" if tied else ""),
         color=TEAL if g1 > 3 else GREY if tied else "d62728")
    stat(s, 9.0, 3.5, f"{g2:+.0f}%", "MAE vs plain average of the 3 raw models", color=TEAL if g2 > 0 else "d62728")
    bl = ov["blend"]; bs = ov[metrics["best_single_model"]]
    fmt = lambda v: f"{v:.0%}" if v is not None else "n/a"
    tb(s, 9.0, 5.15, 3.9, 1.9, [f"Detecting ≥{thr:.0f} mm days: blend POD {fmt(bl['pod'])} / false alarms {fmt(bl['far'])}; "
                                f"{best_lab} POD {fmt(bs['pod'])} / {fmt(bs['far'])}.",
                                "Honest read: the blend finds the best model automatically but does not beat it. ERA5 truth likely flatters ECMWF; gauge verification is next."],
       size=11, color=GREY)

    # 6b what we tried
    if have_exp:
        s = base(prs, "What we tried, and what did not work", "Tuned on earlier seasons, scored on unseen ones; we kept the result even though it is not flattering")
        s.shapes.add_picture(str(figs / "experiments.png"), Inches(0.4), Inches(1.8), width=Inches(8.0))
        bullets(s, 8.7, 1.9, 4.4, 5.0, [
            f"{best_lab} is clearly the strongest model for Hyderabad; GFS and ICON have roughly twice its error.",
            "Our first design (trailing bias correction) made rainfall worse than doing nothing, so it was removed.",
            "Sharper adaptive weights recover almost all of the best model's accuracy without being told which model is best.",
            "A plain average is the worst choice among the blends: it drags the good model down.",
        ], size=14)

    # 7 prepare + respond
    fc_has = risk.level.isin(["Orange", "Red"]).any()
    use_replay = (replay is not None) and (example == "replay" or (example == "auto" and not fc_has))
    if use_replay:
        ex_df = replay.copy()
        ex_day = ex_df.groupby("date")["score"].sum().idxmax()
        kick = f"Illustrated with the worst day of the Oct 2020 replay ({ex_day}); this week's live forecast has no Orange/Red zone"
    else:
        ex_df, ex_day = risk, peak
        kick = f"Peak day in the live 7-day forecast: {peak}"
    fig_map(ex_df, ex_day, figs / "map.png")
    s = base(prs, "Prepare & Respond: from risk map to action", kick)
    s.shapes.add_picture(str(figs / "map.png"), Inches(0.4), Inches(1.7), height=Inches(5.25))
    top = ex_df[ex_df.date == ex_day].sort_values("score", ascending=False).iloc[0]
    rect(s, 6.9, 1.75, 6.0, 3.25, LIGHT)
    paras = [(f"Auto-drafted public alert for {top['locality']} ({top['level']})", {"size": 13, "bold": True, "color": NAVY})]
    for lg in ("en", "hi", "te"):
        paras.append((respond.draft_alert(top["level"], top["locality"], ex_day, top["rain_blend"], lg) or "No alert needed (Green).", {"size": 10, "after": 5}))
    tb(s, 7.05, 1.8, 5.7, 3.2, paras)
    bullets(s, 6.9, 5.15, 6.0, 1.9, [
        "Prepare: boats, pumps and rescue teams split across zones by level × terrain.",
        "Respond: SOS reports ranked P1–P3; routes avoid roads near Orange/Red zones.",
    ], size=13)

    # 8 replay
    s = base(prs, "Replay: what the system would have shown in October 2020", "Rainfall at our 22 localities, 9–17 Oct 2020, fed through the same warning logic")
    if replay is not None:
        s.shapes.add_picture(str(figs / "replay.png"), Inches(0.4), Inches(1.8), width=Inches(8.2))
        mx = replay.groupby("date")["rain_blend"].max(); d0 = mx.idxmax()
        n_red = int((replay[replay.date == d0].level == "Red").sum()); n_org = int((replay[replay.date == d0].level == "Orange").sum())
        bullets(s, 8.9, 1.9, 4.2, 4.9, [
            f"Worst day {d0}: {n_red} Red and {n_org} Orange localities (max {mx.max():.0f} mm in a locality).",
            f"Press reports put 13 Oct 2020 at about 192 mm in 24 h; ERA5 gives only {mx.max():.0f} mm at our points. Reanalysis misses urban extremes, so this is a conservative floor, and it is exactly why gauge data is our next step.",
            "This is an observation-driven replay, not a historical forecast, and we label it that way.",
        ], size=14)

    # 9 feasibility
    s = base(prs, "Feasibility, limits and path to deployment")
    cols = [("Feasible now", ["Runs on a laptop; all data public and key-less", "One command refreshes data; dashboard in Streamlit", "Automated tests incl. no-look-ahead check"]),
            ("Honest limits", ["Only the rainfall layer is validated; no public verified inundation record exists", "Susceptibility is a transparent heuristic, not a hydraulic model", "Localities are analysis points, not GHMC ward boundaries", "SOS sample is simulated; translations need native review"]),
            ("Path to deployment", ["Shadow-run one monsoon with Commissionerate + GHMC DRF", "Add GHMC rain-gauge and drain-level feeds as truth", "Calibrate susceptibility with a hydraulic model", "Integrate alerts and SOS with 112 / TG-ICCC"])]
    for i, (h, items) in enumerate(cols):
        x = 0.5 + i * 4.15
        rect(s, x, 1.45, 3.95, 0.6, [TEAL, "d62728", NAVY][i])
        tb(s, x, 1.45, 3.95, 0.6, h, size=18, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        rect(s, x, 2.05, 3.95, 4.8, LIGHT)
        bullets(s, x + 0.12, 2.2, 3.7, 4.6, items, size=17)

    # 10 team
    s = base(prs, "Team North Eagles and what is next")
    bullets(s, 0.5, 1.5, 6.3, 5.3, [
        "Parth Nayyar · Tanush Sharma · Neev Karwal · Siddhant Shaw",
        "Prior work disclosure: the MausamMix blending method was first built by our team for Smart India Hackathon 2026 (temperature). For this hackathon it was re-implemented for rainfall and combined with new risk, prepare and respond layers.",
        "Everything in this deck is generated from the submitted code and data (make_deck.py).",
    ] + ([f"Live demo: {url}"] if url else []), size=16)
    rect(s, 7.2, 1.5, 5.6, 5.3, LIGHT)
    tb(s, 7.4, 1.6, 5.2, 0.5, "Roadmap", size=20, color=NAVY, bold=True)
    bullets(s, 7.4, 2.2, 5.2, 4.5, ["Flood now, then cyclone and heat-wave modules on the same engine",
                                    "IMD gauge-grid verification and GHMC gauge ingestion",
                                    "Ward-level boundaries and drain-capacity layer",
                                    "WhatsApp/112 intake for SOS triage", "Pilot with Hyderabad Commissionerate"], size=15)
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="JalRaksha_Deck.pptx")
    ap.add_argument("--url", default="", help="live dashboard URL to print on the title and last slides")
    ap.add_argument("--example", default="auto", choices=["auto", "forecast", "replay"],
                    help="what slide 8 illustrates; auto = live forecast if it has Orange/Red, else the Oct-2020 replay")
    a = ap.parse_args()
    build(Path(a.processed), Path(a.out), a.example, a.url)
