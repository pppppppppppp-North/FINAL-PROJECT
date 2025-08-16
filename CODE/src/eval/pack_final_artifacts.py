#!/usr/bin/env python3
from pathlib import Path
import tarfile

def add(tar, p, arc):
    p=Path(p)
    if p.exists():
        tar.add(str(p), arcname=arc)

def main():
    base=Path("CODE/reports")
    out = base/"FINAL_RESULTS.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        add(tar, base/"REPORT.md", "REPORT.md")
        add(tar, base/"auto/REPORT_AUTO.md", "auto/REPORT_AUTO.md")
        for p in base.glob("auto/*.png"):
            add(tar, p, f"auto/{p.name}")
        for p in base.glob("auto/*_preds_ready/*.png"):
            add(tar, p, f"auto/{p.parent.name}/{p.name}")
        for p in base.glob("portfolio/*.png"):
            add(tar, p, f"portfolio/{p.name}")
        for p in base.glob("auto/*.csv"):
            add(tar, p, f"auto/{p.name}")
        for p in base.glob("*.csv"):
            add(tar, p, p.name)
    print(out)
if __name__=="__main__":
    main()
