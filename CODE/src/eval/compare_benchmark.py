#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DATE_CANDS = ["date","ds","timestamp","time"]
CLOSE_CANDS = ["close","Close","adj_close","Adj Close","Adj_Close","AdjClose","close_price","ClosePrice","Last","PX_LAST","Close*","Adj*"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def find_benchmark():
    roots = [Path("CODE/DATA"), Path("DATA"), Path("CODE/data"), Path("data")]
    names = ["SET50","SET_50","SET-50","SET Index","SET","SETINDEX"]
    cands = []
    for r in roots:
        for n in names:
            cands += list(r.glob(f"**/*{n}*.csv"))
    for p in cands:
        try:
            df = pd.read_csv(p)
            return p, df
        except Exception:
            continue
    return None, None

def load_portfolio_series(glob_pat):
    files = sorted(Path(".").glob(glob_pat))
    pnl_list = []
    date_list = []
    for f in files:
        df = pd.read_csv(f)
        if "pnl" not in df.columns:
            continue
        if "date" in df.columns:
            date_list.append(pd.to_datetime(df["date"]).values)
        else:
            date_list.append(np.arange(len(df)))
        pnl_list.append(df["pnl"].astype(float).values)
    if not pnl_list:
        return None, None
    m = min(len(x) for x in pnl_list)
    mat = np.vstack([x[-m:] for x in pnl_list])
    port = mat.mean(axis=0)
    eq = (1.0 + pd.Series(port)).cumprod().values
    dates = date_list[0][-m:]
    return dates, eq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob_tuned", default="CODE/reports/auto/*_preds_ready/backtest_tuned.csv")
    ap.add_argument("--glob_auto", default="CODE/reports/auto/*_preds_ready/backtest_auto.csv")
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    d_tuned, eq_tuned = load_portfolio_series(args.glob_tuned)
    d_auto,  eq_auto  = load_portfolio_series(args.glob_auto)

    bpath, bdf = find_benchmark()
    bench = None
    if bdf is not None:
        dcol = pick_col(bdf, DATE_CANDS)
        ccol = pick_col(bdf, CLOSE_CANDS)
        if ccol is not None:
            b = bdf.copy()
            if dcol:
                b[dcol] = pd.to_datetime(b[dcol])
                b = b.sort_values(dcol)
            r = pd.to_numeric(b[ccol], errors="coerce").pct_change().shift(-1)
            beq = (1.0 + r.fillna(0)).cumprod().values
            bdates = b[dcol].values if dcol else np.arange(len(b))
            bench = (bdates, beq)

    plt.figure()
    if d_tuned is not None:
        plt.plot(pd.to_datetime(d_tuned), eq_tuned, label="Portfolio (tuned)")
    if d_auto is not None:
        plt.plot(pd.to_datetime(d_auto), eq_auto, label="Portfolio (auto)", alpha=0.6)
    if bench is not None:
        bd, be = bench
        try:
            plt.plot(pd.to_datetime(bd), be/be[0], label="Benchmark", alpha=0.8)
        except Exception:
            plt.plot(np.arange(len(be)), be/be[0], label="Benchmark", alpha=0.8)
    plt.title("Portfolio vs Benchmark")
    plt.xlabel("Date"); plt.ylabel("Equity (normalized)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"portfolio_vs_benchmark.png", dpi=180)
    plt.close()
    print(outdir/"portfolio_vs_benchmark.png")
if __name__ == "__main__":
    main()
