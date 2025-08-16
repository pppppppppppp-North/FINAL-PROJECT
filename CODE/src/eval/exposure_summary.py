#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/auto/*_preds_ready/backtest_tuned.csv")
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args = ap.parse_args()
    files = sorted(Path(".").glob(args.glob))
    dfs=[]
    for f in files:
        d = pd.read_csv(f, usecols=[c for c in ["date","position","turnover","pnl"] if c in pd.read_csv(f, nrows=0).columns])
        sym = f.parent.name.replace("_preds_ready","")
        d = d.rename(columns={"position":f"pos_{sym}","turnover":f"to_{sym}","pnl":f"pnl_{sym}"})
        dfs.append(d)
    if not dfs:
        return
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="date", how="outer")
    base = base.sort_values("date")
    pos_cols = [c for c in base.columns if c.startswith("pos_")]
    to_cols  = [c for c in base.columns if c.startswith("to_")]
    base["gross_exposure"] = np.nanmean(np.abs(base[pos_cols].values), axis=1)
    base["net_exposure"]   = np.nanmean(base[pos_cols].values, axis=1)
    base["avg_turnover"]   = np.nanmean(base[to_cols].values, axis=1) if to_cols else 0.0
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base.to_csv(outdir/"exposure_timeseries.csv", index=False)
    plt.figure()
    try:
        plt.plot(pd.to_datetime(base["date"]), base["gross_exposure"], label="Gross")
        plt.plot(pd.to_datetime(base["date"]), base["net_exposure"], label="Net", alpha=0.7)
    except Exception:
        plt.plot(base["gross_exposure"], label="Gross")
        plt.plot(base["net_exposure"], label="Net", alpha=0.7)
    plt.title("Portfolio Exposure")
    plt.xlabel("Date"); plt.ylabel("Exposure")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"exposure.png", dpi=160)
    plt.close()
    print(outdir/"exposure.png")
if __name__=="__main__":
    main()
