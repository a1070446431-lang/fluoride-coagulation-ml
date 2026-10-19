# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [current, current.parent, current.parent.parent]
    for p in candidates:
        if (p / "outputs" / "tables").exists():
            return p
    if current.name.lower() == "src":
        return current.parent
    raise FileNotFoundError("Cannot locate project root.")

PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

print("=" * 100)
print(f"Project root: {PROJECT_ROOT}")
print(f"Table dir   : {TABLE_DIR}")
print("=" * 100)

csv_files = sorted(TABLE_DIR.glob("*.csv"))

keywords = [
    "10_", "11_", "12_", "13_", "14_", "15_", "16_", "17_", "18_", "19_",
    "rf", "catboost", "stack", "weighted", "fusion", "bayes",
    "prediction", "pred", "result", "summary", "metrics"
]

candidate_files = []
for p in csv_files:
    name = p.name.lower()
    if any(k in name for k in keywords):
        candidate_files.append(p)

print(f"[INFO] Candidate csv files found: {len(candidate_files)}")
print()

for p in candidate_files:
    print("-" * 100)
    print(f"FILE: {p.name}")
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(p, encoding="gbk")
        except Exception as e:
            print(f"  [READ FAILED] {e}")
            continue

    print(f"  shape: {df.shape}")
    print("  columns:")
    for c in df.columns:
        print(f"    - {c}")

print("=" * 100)
print("[DONE] Please copy or screenshot the printed file list and columns.")
print("=" * 100)