### What this is
**JalRaksha** ("water protector") turns multi-model rainfall forecasts into flood warnings for Hyderabad localities and
then into the actions a police commissionerate and district disaster cell take: **Predict → Prepare → Respond**.

**Predict.** Three global models (ECMWF IFS, NOAA GFS, DWD ICON) are combined with adaptive, per-locality weights
(MausamMix idea): each model is weighted by (1 / its recent error)^4, using only observations up to two days before the
forecast day (verified by an automated look-ahead test). We also tested a trailing bias correction and **rejected it**:
for bursty rainfall it increased error, so it is switched off. Settings were chosen on seasons before 2025 and the headline
results come from unseen 2025-26 monsoons. Warning levels follow IMD 24-hour rainfall classes (Yellow >=15.6 mm,
Orange >=64.5 mm, Red >=115.6 mm), after scaling rainfall by terrain susceptibility (+/-25%).

**Prepare.** Pre-positions rescue teams, boats and pumps across Yellow/Orange/Red zones and shows first-alert dates.

**Respond.** Drafts public alerts in English/Hindi/Telugu, triages citizen SOS reports, and computes flood-aware routes
over the OpenStreetMap major-road network.

### Honest limitations
- **Validation covers the rainfall layer only.** There is no public, verified inundation record to test the ward-level
  flood score against; terrain susceptibility is a transparent heuristic, not a calibrated hydrological model.
- **Truth data is ERA5-based reanalysis**, which comes from ECMWF's own system (likely flattering ECMWF forecasts), smooths and under-reports extreme urban rainfall. Gauge-based IMD
  grids / GHMC rain-gauge (TSDPS) data would be the right production truth.
- **Localities are analysis points, not GHMC ward boundaries.** Coordinates are approximate.
- In our tests ECMWF IFS clearly outperforms GFS and ICON for Hyderabad; the blend matches it on error but is not better at detecting heavy-rain days.
- Weights are learned on day-1-lead history and applied to the whole 7-day horizon; skill falls with lead time, which
  the *confidence* column (model disagreement) partly reflects.
- The sample SOS reports are **simulated** for demonstration. Alert translations need native-speaker review.
- Routing penalises roads near Orange/Red localities; it does not know live road closures.

### Path to deployment
Shadow-run one monsoon with the Hyderabad Commissionerate/GHMC DRF, add GHMC rain-gauge and drain-level feeds, replace
the susceptibility heuristic with a calibrated hydrological/hydraulic layer, and integrate alerts with the 112 / TG-ICCC systems.
