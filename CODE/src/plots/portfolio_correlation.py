#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/*/backtest_preds_wf_tuned.csv")
    ap.add_argument("--outdir", default="CODE/reports/portfolio")
    args = ap.parse_args()
    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    dfs=[]
    for f in files:
        sym = Path(f).parent.name
        df = pd.read_csv(f, usecols=["date","pnl"])
        df["date"]=pd.to_datetime(df["date"])
        df = df.rename(columns={"pnl":sym})
        dfs.append(df)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="date", how="outer")
    base = base.sort_values("date").set_index("date")
    ret = base.fillna(0.0)
    corr = ret.corr()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    corr.to_csv(Path(args.outdir)/"pnl_correlation.csv")
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("P&L Correlation Between Symbols")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/"pnl_correlation.png", dpi=180)
    plt.close()
if __name__=="__main__":
    main()
