"""One command: fetch real data -> pipeline -> deck.   python run_all.py [--roads]"""
import subprocess, sys
roads = ["--roads"] if "--roads" in sys.argv else []
for cmd in ([sys.executable, "fetch_data.py"] + roads, [sys.executable, "run_pipeline.py"], [sys.executable, "make_deck.py"]):
    print("\n>>", " ".join(cmd)); r = subprocess.run(cmd)
    if r.returncode: sys.exit(r.returncode)
print("\nAll done. Launch the dashboard with: streamlit run app.py")
