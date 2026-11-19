# JalRaksha — Flood early-warning & response for Hyderabad

**CipherCop 2026 · Track 2: AI for Disaster Resilience · Team North Eagles**
(Parth Nayyar, Tanush Sharma, Neev Karwal, Siddhant Shaw)

JalRaksha turns multi-model rainfall forecasts into flood warnings for Hyderabad localities, then into the actions a
police commissionerate and district disaster cell take: **Predict → Prepare → Respond**.

| Layer | What it does |
|---|---|
| **Predict** | Combines ECMWF IFS, NOAA GFS and DWD ICON rainfall with *MausamMix*-style adaptive per-locality weights ((1/recent error)^4, no look-ahead). A trailing bias correction was tested and rejected (it hurt rainfall); `experiments.py` reproduces that comparison. Warning level (Green/Yellow/Orange/Red) from IMD rainfall classes, scaled by terrain susceptibility. |
| **Prepare** | Allocates rescue teams, boats and pumps across at-risk localities; shows first-alert dates over a 7-day horizon. |
| **Respond** | Multilingual (EN/HI/TE) alert drafts; SOS-report triage (P1/P2/P3); flood-aware routing over OpenStreetMap major roads. |

Prior work disclosure: the MausamMix blending method was first developed by our team for Smart India Hackathon 2026
(2 m temperature). Here it is re-implemented and extended to rainfall and combined with new risk, prepare and respond layers.

## Live demo
See the link in the presentation (also deployed on Streamlit Community Cloud from this repository; main file `app.py`).

## Quick start (about 10 minutes, needs internet once)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python fetch_data.py --roads     # downloads the real datasets into data/raw/ (drop --roads to skip routing; --roads needs `pip install osmnx`)
python experiments.py            # optional: tuning vs unseen-holdout comparison of blend settings
python run_pipeline.py           # blend + validation + risk tables -> data/processed/
streamlit run app.py             # dashboard at http://localhost:8501
python make_deck.py              # optional: regenerates the presentation from the real results
```

Google Colab works too: upload the folder, `!pip install -r requirements.txt`, run the first two commands, then
`!streamlit run app.py & npx localtunnel --port 8501` (or just inspect `data/processed/*.csv` and `metrics.json`).

## Data (all free, no API key) — see `dataset/data_sources_links.txt`
- Open-Meteo Forecast API (live 7-day, three models) · Historical Forecast API (past model forecasts)
- Open-Meteo Historical Weather API (ERA5-based reanalysis; used as observation truth)
- Open-Meteo Elevation API (Copernicus DEM) for terrain susceptibility
- OpenStreetMap major roads (optional, via osmnx)
- `data/demo_sos_reports_SIMULATED.csv` — **simulated** citizen reports for the triage demo only

`fetch_data.py` writes everything to `data/raw/`. Those CSVs are the dataset to submit.

## Tests
```bash
PYTHONPATH=. python -m pytest -q tests
```
Covers: no-look-ahead guarantee of the blend, bias-correction benefit, weights sum to 1, IMD thresholds, allocation
totals, triage ordering, alerts in all languages, flood-aware routing, and a headless run of the Streamlit app.
(`tests/make_fixture.py` creates *synthetic test data only*; it is never used by the app unless you point it there.)

## Layout
```
fetch_data.py        download real data          jalraksha/blend.py    MausamMix rainfall blending
run_pipeline.py      backtest + risk tables      jalraksha/risk.py     susceptibility, levels, confidence
app.py               Streamlit dashboard         jalraksha/respond.py  allocation, triage, alerts, routing
make_deck.py         builds the deck from real   docs/ABOUT.md         method + limitations
tests/               pytest suite
```

## Limitations (also shown in the app)
Validation covers the rainfall layer only (no public verified inundation record exists to test ward flood scores);
truth is ERA5-based reanalysis, which under-reports extreme rain; localities are analysis points not GHMC wards; the
susceptibility score is a transparent heuristic, not a hydraulic model; SOS data is simulated; translations need
native-speaker review.
