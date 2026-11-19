"""Smoke-test the Streamlit app headlessly on TEST FIXTURE data (never real/ship data)."""
import os
import networkx as nx
from streamlit.testing.v1 import AppTest

import run_pipeline
from tests.make_fixture import make
from pathlib import Path


def test_app_runs(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    make(raw)
    G = nx.MultiDiGraph()
    for i in range(30):
        for j in range(30):
            G.add_node(f"{i}_{j}", y=17.30 + i * 0.0075, x=78.35 + j * 0.0075)
    for i in range(30):
        for j in range(30):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < 30 and j + dj < 30:
                    G.add_edge(f"{i}_{j}", f"{i+di}_{j+dj}", length=800.0)
                    G.add_edge(f"{i+di}_{j+dj}", f"{i}_{j}", length=800.0)
    nx.write_graphml(G, raw / "hyderabad_roads.graphml")
    run_pipeline.main(raw, proc)
    monkeypatch.setenv("JALRAKSHA_PROCESSED", str(proc))
    monkeypatch.setenv("JALRAKSHA_RAW", str(raw))
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) >= 9
    # change day + language and rerun
    at.sidebar.select_slider[0].set_value(at.sidebar.select_slider[0].options[3]).run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio[1].set_value("te").run()            # radio[0] is Scenario, radio[1] is alert language
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio[0].set_value("Replay: Oct 2020 floods").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Red zones" in m.label for m in at.metric)
