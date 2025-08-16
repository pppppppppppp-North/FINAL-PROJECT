#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--out_csv", default="CODE/reports/data_leakage_checks.csv")
    args = ap.parse_args()
    rows=[]
    for f in sorted(Path(".").glob(args.glob)):
        sym = Path(f).stem.split("_test_preds")[0]
        df = pd.read_csv(f)
        has_date = any(c in df.columns for c in ["date","ds","timestamp","time"])
        has_future = any(("t+1" in c.lower() or "lead" in c.lower() or "future" in c.lower()) for c in df.columns)
        dup = df.duplicated().any()
        n = len(df)
        null_rate = float(df.isna().mean().mean())
        rows.append({"symbol":sym,"rows":n,"has_date":has_date,"has_future_colname":has_future,"has_dup_rows":bool(dup),"mean_null_rate":null_rate})
    out = pd.DataFrame(rows).sort_values("symbol")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(args.out_csv)
if __name__=="__main__":
    main()
