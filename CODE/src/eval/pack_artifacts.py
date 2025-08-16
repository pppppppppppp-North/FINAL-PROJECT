#!/usr/bin/env python3
from pathlib import Path
import tarfile

def add_if_exists(tar, path, arc):
    p = Path(path)
    if p.exists():
        tar.add(str(p), arcname=arc)

def main():
    base = Path("CODE/reports")
    out = base/"FINAL_PROJECT_artifacts.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        add_if_exists(tar, base/"REPORT.md", "REPORT.md")
        for p in base.glob("backtest_summary_*.csv"):
            add_if_exists(tar, p, p.name)
        for p in base.glob("portfolio/*.png"):
            add_if_exists(tar, p, f"portfolio/{p.name}")
        for p in base.glob("portfolio/*.csv"):
            add_if_exists(tar, p, f"portfolio/{p.name}")
        for d in base.glob("*/"):
            for q in d.glob("*.png"):
                add_if_exists(tar, q, f"{d.name}/{q.name}")
            for q in d.glob("*.json"):
                add_if_exists(tar, q, f"{d.name}/{q.name}")
            for q in d.glob("backtest_*.csv"):
                add_if_exists(tar, q, f"{d.name}/{q.name}")
    print(out)
if __name__=="__main__":
    main()
