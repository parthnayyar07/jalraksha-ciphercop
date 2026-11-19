"""Static configuration for JalRaksha (Hyderabad flood early-warning + response)."""

# Approximate centroid coordinates of well-known Hyderabad localities (public map knowledge).
# These are analysis points, NOT official GHMC ward boundaries.
LOCALITIES = [
    ("Charminar", 17.3616, 78.4747),
    ("Chaderghat", 17.3830, 78.4900),
    ("Malakpet", 17.3750, 78.5100),
    ("Amberpet", 17.3920, 78.5160),
    ("Musheerabad", 17.4170, 78.4950),
    ("Tank Bund / Hussain Sagar", 17.4239, 78.4738),
    ("Begumpet", 17.4440, 78.4676),
    ("Secunderabad", 17.4399, 78.4983),
    ("Mehdipatnam", 17.3950, 78.4350),
    ("Tolichowki", 17.4000, 78.4100),
    ("Attapur", 17.3680, 78.4260),
    ("Falaknuma", 17.3300, 78.4680),
    ("Dilsukhnagar", 17.3688, 78.5247),
    ("Saroornagar", 17.3541, 78.5360),
    ("LB Nagar", 17.3457, 78.5522),
    ("Nagole", 17.3920, 78.5640),
    ("Uppal", 17.4058, 78.5591),
    ("Kukatpally", 17.4948, 78.3996),
    ("Miyapur", 17.4967, 78.3560),
    ("Madhapur", 17.4486, 78.3908),
    ("Gachibowli", 17.4401, 78.3489),
    ("Alwal", 17.5000, 78.5100),
]

# Approximate location of the police command centre (Banjara Hills). Editable in the app.
COMMAND_CENTRE = ("Command centre (approx.)", 17.4126, 78.4482)

# Weather models served by Open-Meteo that we blend.
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
MODEL_LABELS = {
    "ecmwf_ifs025": "ECMWF IFS",
    "gfs_seamless": "NOAA GFS",
    "icon_seamless": "DWD ICON",
}

# History window used to learn bias + weights and to validate (monsoon-heavy, 3 seasons).
HISTORY_START = "2022-06-01"
HISTORY_END = None  # None -> yesterday minus archive lag (resolved in fetch.py)

# IMD 24-h rainfall classes (mm): heavy 64.5-115.5, very heavy 115.6-204.4, extremely heavy >=204.5
IMD_MODERATE = 15.6
IMD_HEAVY = 64.5
IMD_VERY_HEAVY = 115.6
IMD_EXTREME = 204.5

# Event threshold used for validation of rain detection (mm/day).
EVENT_THRESHOLD_MM = 20.0

BIAS_WINDOW_DAYS = 30
BIAS_LAG_DAYS = 2  # forecast for day t may only use observations up to t-2 (no look-ahead)
BIAS_RATIO_CLIP = (0.5, 2.0)
RATIO_POWER = 0.0     # chosen on tuning seasons (<2025): bias correction made rainfall WORSE, so it is off (1 = full, 0 = none)
WEIGHT_POWER = 4.0    # chosen on tuning seasons (<2025): weights ~ (1/MAE)^4 so the clearly-best model dominates
HOLDOUT_FROM_YEAR = 2025  # headline metrics use monsoon seasons from this year on (never used for tuning)

LEVEL_COLORS = {
    "Green": [46, 160, 67],
    "Yellow": [240, 200, 0],
    "Orange": [245, 130, 20],
    "Red": [214, 39, 40],
}
LEVEL_ORDER = ["Green", "Yellow", "Orange", "Red"]

# Famous test case for the replay: the October 2020 Hyderabad floods.
REPLAY_START = "2020-10-09"
REPLAY_END = "2020-10-17"
