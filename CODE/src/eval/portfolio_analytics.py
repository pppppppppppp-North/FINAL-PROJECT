#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def rolling_max_drawdown(equity, window=252):
    eq = pd.Series(equity).astype(float)
    roll_max = eq.rolling(window, min_periods=1).max()
    dd = eq/roll_max - 1.0
    return dd

def rolling_sharpe(returns, window=126):
    r = pd.Series(returns).astype(float)
    vol = r.rolling(window, min_periods=1).std(ddof=1)
    mean = r.rolling(window, min_periods=1).mean()
    out = (mean / (vol.replace(0, np.nan))) * np.sqrt(252)
    return out.fillna(0.0)

def align_and_stack(csv_paths, date_col="date"):
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        else:
            df[date_col] = pd.RangeIndex(len(df))
        sym = Path(p).parent.name
        df = df[[date_col, "pnl", "equity"]].copy()
        df.columns = [date_col, f"pnl_{sym}", f"eq_{sym}"]
        dfs.append(df)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on=date_col, how="outer")
    base = base.sort_values(date_col).reset_index(drop=True)
    pnl_cols = [c for c in base.columns if c.startswith("pnl_")]
    eq_cols  = [c for c in base.columns if c.startswith("eq_")]
    pnl_mat = base[pnl_cols].to_numpy(dtype=float)
    port_pnl = np.nanmean(pnl_mat, axis=1)
    port_eq = (1.0 + pd.Series(port_pnl).fillna(0.0)).cumprod().values
    base["port_pnl"] = port_pnl
    base["port_eq"] = port_eq
    return base, pnl_cols, eq_cols

def metrics_from_pnl(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return {"CAGR":0.0,"Sharpe":0.0,"MaxDD":0.0,"WinRate":0.0,"FinalEquity":1.0}
    eq = (1.0 + pnl).cumprod()
    sharpe = 0.0 if pnl.std(ddof=1)==0 else pnl.mean()/pnl.std(ddof=1)*np.sqrt(252)
    cagr = eq[-1]**(252/max(len(eq),1)) - 1.0
    roll_max = np.maximum.accumulate(eq)
    mdd = float(np.min(eq/roll_max - 1.0))
    win = float(np.mean(pnl>0))
    return {"CAGR":float(cagr),"Sharpe":float(sharpe),"MaxDD":mdd,"WinRate":win,"FinalEquity":float(eq[-1])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/*/backtest_preds_wf_tuned.csv")
    ap.add_argument("--outdir", default="CODE/reports/portfolio")
    ap.add_argument("--roll_sharpe_win", type=int, default=126)
    ap.add_argument("--roll_dd_win", type=int, default=252)
    args = ap.parse_args()

    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df, pnl_cols, eq_cols = align_and_stack(files)
    df.to_csv(Path(args.outdir)/"portfolio_timeseries.csv", index=False)

    port_metrics = metrics_from_pnl(df["port_pnl"].fillna(0.0).values)
    with open(Path(args.outdir)/"portfolio_metrics.json","w") as fh:
        json.dump(port_metrics, fh, indent=2)

    plt.figure()
    plt.plot(df["date"], df["port_eq"])
    plt.title("Portfolio Equity (equal-weight)")
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/"portfolio_equity.png", dpi=160)
    plt.close()

    rsh = rolling_sharpe(df["port_pnl"].fillna(0.0).values, window=args.roll_sharpe_win)
    plt.figure()
    plt.plot(df["date"], rsh.values)
    plt.title(f"Rolling Sharpe ({args.roll_sharpe_win}d)")
    plt.xlabel("Date"); plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/"rolling_sharpe.png", dpi=160)
    plt.close()

    rdd = rolling_max_drawdown(df["port_eq"].values, window=args.roll_dd_win)
    plt.figure()
    plt.plot(df["date"], rdd.values)
    plt.title(f"Rolling Max Drawdown ({args.roll_dd_win}d)")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/"rolling_drawdown.png", dpi=160)
    plt.close()

    sym_sharpe = []
    for c in pnl_cols:
        pnl = df[c].fillna(0.0).values
        m = metrics_from_pnl(pnl)
        sym = c.replace("pnl_","")
        sym_sharpe.append((sym, m["Sharpe"], m["CAGR"], m["FinalEquity"]))
    rank = pd.DataFrame(sym_sharpe, columns=["symbol","Sharpe","CAGR","FinalEquity"]).sort_values("Sharpe", ascending=False)
    rank.to_csv(Path(args.outdir)/"symbol_ranking.csv", index=False)

if __name__ == "__main__":
    main()
