#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def backtest(df, task, thr, allow_short, tc_bps):
    if "date" in df.columns:
        df = df.sort_values("date")
    if task == "classification":
        t = max(thr, 0.5)
        signal = (df["y_pred"] >= t).astype(int) - ((df["y_pred"] < (1-t)).astype(int) if allow_short else 0)
        if "ret" in df.columns:
            ret = df["ret"].astype(float).values
        else:
            ret = np.where(df["y_true"].astype(int)>0,1.0,-1.0)
    else:
        signal = np.where(df["y_pred"] > thr, 1, np.where(df["y_pred"] < -thr, -1, 0))
        ret = (df["ret"] if "ret" in df.columns else df["y_true"]).astype(float).values
    pos = signal.astype(float)
    pos_prev = np.roll(pos,1); pos_prev[0]=0.0
    turnover = np.abs(pos-pos_prev)
    tc = turnover*(tc_bps/1e4)
    pnl = pos*ret - tc
    df["signal"]=signal; df["position"]=pos; df["turnover"]=turnover; df["tc"]=tc; df["pnl"]=pnl
    df["equity"] = (1.0 + df["pnl"]).cumprod()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--pred_col", required=True)
    ap.add_argument("--true_col", default="")
    ap.add_argument("--ret_col", default="")
    ap.add_argument("--date_col", default="")
    ap.add_argument("--task", choices=["auto","classification","regression"], default="auto")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--allow_short", type=int, default=0)
    ap.add_argument("--tc_bps", type=float, default=10)
    ap.add_argument("--tag", default="explicit")
    ap.add_argument("--outdir", default="CODE/reports")
    args = ap.parse_args()

    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    for f in files:
        df = pd.read_csv(f)
        cols = {}
        if args.date_col and args.date_col in df.columns:
            df = df.rename(columns={args.date_col:"date"})
        if args.true_col and args.true_col in df.columns:
            df = df.rename(columns={args.true_col:"y_true"})
        if args.ret_col and args.ret_col in df.columns:
            df = df.rename(columns={args.ret_col:"ret"})
        df = df.rename(columns={args.pred_col:"y_pred"})
        sym = Path(f).stem.split("_test_preds")[0]
        if args.task=="auto":
            task = "classification" if ("y_true" in df.columns and set(pd.unique(df["y_true"].dropna().astype(int))).issubset({0,1})) else "regression"
        else:
            task = args.task
        bt = backtest(df, task, args.thr, args.allow_short, args.tc_bps)
        outdir = Path(args.outdir)/sym
        outdir.mkdir(parents=True, exist_ok=True)
        bt.to_csv(outdir/f"backtest_{args.tag}.csv", index=False)
        plt.figure()
        x = pd.to_datetime(bt["date"]) if "date" in bt.columns else range(len(bt))
        plt.plot(x, bt["equity"])
        plt.title(f"Equity: {sym}")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(outdir/f"equity_{args.tag}.png", dpi=150)
        plt.close()
        with open(outdir/f"backtest_{args.tag}_meta.json","w") as fh:
            json.dump({"symbol":sym,"task":task,"thr":args.thr,"tc_bps":args.tc_bps,"pred_col":args.pred_col,"true_col":args.true_col,"ret_col":args.ret_col,"date_col":args.date_col}, fh, indent=2)
if __name__=="__main__":
    main()
