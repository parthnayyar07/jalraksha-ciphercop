"""Prepare + Respond layers: resource allocation, SOS triage, multilingual alerts, safe routing."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- PREPARE: allocation
LEVEL_WEIGHT = {"Green": 0.0, "Yellow": 1.0, "Orange": 2.0, "Red": 4.0}


def allocate(risk_day: pd.DataFrame, resources: dict[str, int]) -> pd.DataFrame:
    """Split each resource type across localities in proportion to level weight x (0.5+susceptibility).

    Largest-remainder rounding keeps totals exact; every Red zone gets at least one of each
    resource while stock lasts.
    """
    df = risk_day[["locality", "level", "susc"]].copy()
    df["w"] = df["level"].map(LEVEL_WEIGHT) * (0.5 + df["susc"].fillna(0.5))
    out = pd.DataFrame({"locality": df["locality"], "level": df["level"]})
    for res, total in resources.items():
        alloc = np.zeros(len(df), dtype=int)
        if total > 0 and df["w"].sum() > 0:
            red = np.where(df["level"].values == "Red")[0]
            remaining = total
            for i in red:
                if remaining > 0:
                    alloc[i] += 1
                    remaining -= 1
            if remaining > 0:
                share = df["w"].values / df["w"].sum() * remaining
                base = np.floor(share).astype(int)
                alloc += base
                left = remaining - base.sum()
                order = np.argsort(-(share - base))
                for i in order[:left]:
                    if df["w"].values[i] > 0:
                        alloc[i] += 1
        out[res] = alloc
    return out


# ---------------------------------------------------------------- RESPOND: SOS triage
KEYWORDS = [
    (r"drown|sinking|swept|carried away|डूब|మునిగ", 40, "drowning risk"),
    (r"trapped|stuck inside|cannot get out|फंसे|फँसे|చిక్కుకు", 30, "people trapped"),
    (r"collapse|wall fell|roof fell", 30, "structural collapse"),
    (r"electric|current|shock|live wire|करंट|కరెంట్", 25, "electrocution hazard"),
    (r"pregnan|labou?r pain", 20, "pregnant woman"),
    (r"oxygen|dialysis|insulin|ambulance|hospital|medical|heart|unconscious", 20, "medical emergency"),
    (r"child|children|baby|infant|kid|बच्चा|बच्चे|పిల్ల", 15, "children present"),
    (r"elderly|old man|old woman|senior|disabled|बुज़ुर्ग|बुजुर्ग|వృద్ధ", 15, "elderly/disabled"),
    (r"no food|no water|stranded|marooned|no power", 8, "stranded / no supplies"),
]
ZONE_BONUS = {"Red": 15, "Orange": 8, "Yellow": 3, "Green": 0}


def triage(reports: pd.DataFrame, risk_day: pd.DataFrame) -> pd.DataFrame:
    """reports columns: report_id, locality, text, people (int), depth_cm (float, optional)."""
    lvl = dict(zip(risk_day["locality"], risk_day["level"]))
    rows = []
    for _, r in reports.iterrows():
        text = str(r.get("text", "")).lower()
        score, reasons = 0.0, []
        for pat, pts, why in KEYWORDS:
            if re.search(pat, text):
                score += pts
                reasons.append(why)
        people = int(r.get("people", 0) or 0)
        if people:
            score += min(people, 10) * 2
            reasons.append(f"{people} people")
        depth = float(r.get("depth_cm", 0) or 0)
        if depth:
            score += min(depth, 150) / 5
            reasons.append(f"water ~{depth:.0f} cm")
        zone = lvl.get(r.get("locality"), "Green")
        score += ZONE_BONUS[zone]
        if zone != "Green":
            reasons.append(f"{zone} zone")
        pr = "P1 - immediate" if score >= 60 else "P2 - urgent" if score >= 35 else "P3 - monitor"
        rows.append({**r.to_dict(), "score": round(score, 1), "priority": pr, "why": "; ".join(reasons)})
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False).reset_index(drop=True) if len(out) else out


# ---------------------------------------------------------------- RESPOND: alerts
ALERTS = {
    "Red": {
        "en": "RED ALERT: Very heavy rain (about {rain} mm) expected in {loc} on {date}, with flood risk. "
              "People in low-lying areas should move to safer ground now. Do not walk or drive through "
              "flood water. Emergency: call 100 / 112.",
        "hi": "रेड अलर्ट: {date} को {loc} क्षेत्र में अत्यधिक भारी बारिश (लगभग {rain} मिमी) और बाढ़ का खतरा है। "
              "निचले इलाकों के लोग तुरंत सुरक्षित स्थान पर जाएँ। बाढ़ के पानी में न चलें और न वाहन चलाएँ। "
              "आपात स्थिति में 100 / 112 पर कॉल करें।",
        "te": "ఎరుపు హెచ్చరిక: {date} న {loc} ప్రాంతంలో అతి భారీ వర్షం (సుమారు {rain} మి.మీ.) మరియు వరద ముప్పు ఉంది. "
              "లోతట్టు ప్రాంతాల వారు వెంటనే సురక్షిత ప్రదేశాలకు వెళ్లండి. వరద నీటిలో నడవకండి లేదా వాహనం నడపకండి. "
              "అత్యవసరానికి 100 / 112 కు కాల్ చేయండి.",
    },
    "Orange": {
        "en": "ORANGE ALERT: Heavy rain (about {rain} mm) expected in {loc} on {date}. Waterlogging likely. "
              "Avoid underpasses and low-lying roads; keep valuables and vehicles on higher ground. "
              "Emergency: call 100 / 112.",
        "hi": "ऑरेंज अलर्ट: {date} को {loc} क्षेत्र में भारी बारिश (लगभग {rain} मिमी) की संभावना है। जलभराव हो सकता है। "
              "अंडरपास और निचली सड़कों से बचें; सामान और वाहन ऊँची जगह रखें। आपात स्थिति में 100 / 112 पर कॉल करें।",
        "te": "నారింజ హెచ్చరిక: {date} న {loc} ప్రాంతంలో భారీ వర్షం (సుమారు {rain} మి.మీ.) పడే అవకాశం ఉంది. "
              "నీటి నిల్వ ఏర్పడవచ్చు. అండర్‌పాస్‌లు మరియు లోతట్టు రోడ్లను నివారించండి. "
              "అత్యవసరానికి 100 / 112 కు కాల్ చేయండి.",
    },
    "Yellow": {
        "en": "YELLOW ADVISORY: Moderate to heavy rain (about {rain} mm) possible in {loc} on {date}. "
              "Be aware of local waterlogging and plan travel accordingly.",
        "hi": "येलो एडवाइज़री: {date} को {loc} क्षेत्र में मध्यम से भारी बारिश (लगभग {rain} मिमी) संभव है। "
              "स्थानीय जलभराव से सावधान रहें और यात्रा की योजना उसी अनुसार बनाएँ।",
        "te": "పసుపు సూచన: {date} న {loc} ప్రాంతంలో మోస్తరు నుండి భారీ వర్షం (సుమారు {rain} మి.మీ.) పడవచ్చు. "
              "స్థానిక నీటి నిల్వ పట్ల జాగ్రత్తగా ఉండి ప్రయాణాన్ని తదనుగుణంగా ప్లాన్ చేసుకోండి.",
    },
}
LANGS = {"en": "English", "hi": "हिन्दी", "te": "తెలుగు"}


def draft_alert(level: str, loc: str, date: str, rain: float, lang: str) -> str:
    if level not in ALERTS:
        return ""
    return ALERTS[level][lang].format(loc=loc, date=date, rain=int(round(rain)))


# ---------------------------------------------------------------- RESPOND: safe routing
def _hav_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


class RoadNet:
    """Light wrapper around a networkx graph with node attrs x (lon), y (lat), edge attr length (m)."""

    def __init__(self, G):
        import networkx as nx
        self.nx = nx
        H = nx.Graph()
        for u, v, d in G.edges(data=True):
            L = float(d.get("length", 1.0))
            if H.has_edge(u, v) and H[u][v]["length"] <= L:
                continue
            H.add_edge(u, v, length=L)
        for n, d in G.nodes(data=True):
            if n in H:
                H.nodes[n]["x"] = float(d["x"])
                H.nodes[n]["y"] = float(d["y"])
        self.G = H
        self.nodes = list(H.nodes)
        self.xy = np.array([[H.nodes[n]["y"], H.nodes[n]["x"]] for n in self.nodes])

    @classmethod
    def load(cls, path: str | Path):
        import networkx as nx
        return cls(nx.read_graphml(path))

    def nearest(self, lat: float, lon: float):
        d = _hav_km(lat, lon, self.xy[:, 0], self.xy[:, 1])
        return self.nodes[int(np.argmin(d))]

    def _edge_penalties(self, zones: pd.DataFrame, radius_km: float, pen: dict):
        """Penalty multiplier per edge: roads whose midpoint lies near Orange/Red zones."""
        bad = zones[zones["level"].isin(["Orange", "Red"])]
        mult = {}
        for u, v in self.G.edges:
            my = (self.G.nodes[u]["y"] + self.G.nodes[v]["y"]) / 2
            mx = (self.G.nodes[u]["x"] + self.G.nodes[v]["x"]) / 2
            m = 1.0
            for _, z in bad.iterrows():
                if _hav_km(my, mx, z["lat"], z["lon"]) <= radius_km:
                    m = max(m, pen[z["level"]])
            mult[(u, v)] = m
        return mult

    def route(self, origin, dest, zones: pd.DataFrame, radius_km: float = 1.5,
              pen: dict | None = None):
        """Returns dict with plain shortest path and flood-aware path (coords, km, flagged km)."""
        pen = pen or {"Orange": 4.0, "Red": 25.0}
        o, d = self.nearest(*origin), self.nearest(*dest)
        mult = self._edge_penalties(zones, radius_km, pen)
        nx = self.nx

        def w_plain(u, v, e):
            return e["length"]

        def w_safe(u, v, e):
            return e["length"] * mult.get((u, v), mult.get((v, u), 1.0))

        out = {}
        for key, w in (("direct", w_plain), ("safe", w_safe)):
            path = nx.shortest_path(self.G, o, d, weight=w)
            km, flagged = 0.0, 0.0
            for a, b in zip(path[:-1], path[1:]):
                L = self.G[a][b]["length"] / 1000
                km += L
                if mult.get((a, b), mult.get((b, a), 1.0)) > 1.0:
                    flagged += L
            out[key] = {
                "coords": [[self.G.nodes[n]["x"], self.G.nodes[n]["y"]] for n in path],
                "km": km,
                "flooded_km": flagged,
            }
        return out
