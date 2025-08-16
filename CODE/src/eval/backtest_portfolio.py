#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def backtest_one(df, task, thr, allow_short, tc_bps):
    if "date" in df.columns:
        df = df.sort_values("date")
    if task == "classification":
        t = max(thr, 0.5)
        long_sig = (df["y_pred"] >= t).astype(int)
        short_sig = (df["y_pred"] < (1 - t)).astype(int) if allow_short else 0
        signal = long_sig - short_sig
        if "ret" in df.columns:
            ret = df["ret"].astype(float).values
        elif "y_true" in df.columns:
            ret = np.where(df["y_true"].astype(int) > 0, 1.0, -1.0)
        else:
            raise ValueError("need ret or y_true")
    else:
        signal = np.where(df["y_pred"] > thr, 1, np.where(df["y_pred"] < -thr, -1, 0))
        ret = (df["ret"] if "ret" in df.columns else df["y_true"]).astype(float).values
    pos = signal.astype(float)
    pos_prev = np.roll(pos, 1); pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    tc = turnover * (tc_bps / 1e4)
    pnl = pos * ret - tc
    df["signal"] = signal
    df["position"] = pos
    df["turnover"] = turnover
    df["tc"] = tc
    df["pnl"] = pnl
    df["equity"] = (1.0 + df["pnl"]).cumprod()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ret/*_test_preds.csv")
    ap.add_argument("--pred_col", required=True)
    ap.add_argument("--ret_col", default="")
    ap.add_argument("--date_col", default="")
    ap.add_argument("--true_col", default="")
    ap.add_argument("--task", choices=["regression","classification"], default="regression")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--allow_short", type=int, default=0)
    ap.add_argument("--tc_bps", type=float, default=10)
    ap.add_argument("--tag", default="auto")
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args = ap.parse_args()

    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    rows = []
    pnl_list = []
    date_list = []

    for f in files:
        df = pd.read_csv(f)
        if args.date_col and args.date_col in df.columns:
            df = df.rename(columns={args.date_col: "date"})
        if args.ret_col and args.ret_col in df.columns:
            df = df.rename(columns={args.ret_col: "ret"})
        if args.true_col and args.true_col in df.columns:
            df = df.rename(columns={args.true_col: "y_true"})
        df = df.rename(columns={args.pred_col: "y_pred"})
        sym = infer_symbol(f)

        bt = backtest_one(df.copy(), args.task, args.thr, args.allow_short, args.tc_bps)

        outdir_sym = Path(args.outdir) / sym
        outdir_sym.mkdir(parents=True, exist_ok=True)
        bt.to_csv(outdir_sym / f"backtest_{args.tag}.csv", index=False)

        plt.figure()
        x = pd.to_datetime(bt["date"]) if "date" in bt.columns else range(len(bt))
        plt.plot(x, bt["equity"])
        plt.title(f"Equity: {sym}")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(outdir_sym / f"equity_{args.tag}.png", dpi=160)
        plt.close()

        ret = bt["pnl"].values
        eq = (1.0 + ret).cumprod()
        sharpe = 0.0 if ret.std(ddof=1)==0 else ret.mean()/ret.std(ddof=1)*np.sqrt(252)
        rows.append({"symbol": sym, "Sharpe": float(sharpe), "FinalEquity": float(eq[-1])})
        pnl_list.append(ret)
        date_list.append(bt["date"].values if "date" in bt.columns else np.arange(len(bt)))

    if rows:
        pd.DataFrame(rows).sort_values("symbol").to_csv(Path(args.outdir)/f"summary_{args.tag}.csv", index=False)
        m = min(len(x) for x in pnl_list)
        if m > 0:
            mat = np.vstack([x[-m:] for x in pnl_list])
            port = mat.mean(axis=0)
            peq = (1.0 + pd.Series(port)).cumprod().values
            try:
                dates = pd.to_datetime(date_list[0][-m:])
            except Exception:
                dates = np.arange(m)
            plt.figure()
            plt.plot(dates, peq)
            plt.title("Portfolio Equity")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(Path(args.outdir)/f"portfolio_equity_{args.tag}.png", dpi=180)
            plt.close()
            port_sharpe = 0.0 if port.std(ddof=1)==0 else port.mean()/port.std(ddof=1)*np.sqrt(252)
            json.dump({"Sharpe": float(port_sharpe), "FinalEquity": float(peq[-1])}, open(Path(args.outdir)/f"portfolio_metrics_{args.tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()
