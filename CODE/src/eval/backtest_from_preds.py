#!/usr/bin/env python3
import argparse, re, math, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

CAND_TRUE = ["y_true","true","label","target","y","actual","direction"]
CAND_PRED = ["y_pred","pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]
CAND_RET  = ["ret","return","returns","y_ret","next_ret","r","y_next"]

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def detect_cols(df):
    y_true = None; y_pred = None; date = None; ret = None
    for c in CAND_TRUE:
        if c in df.columns: y_true = c; break
    for c in CAND_PRED:
        if c in df.columns: y_pred = c; break
    for c in ["date","timestamp","time","ds"]:
        if c in df.columns: date = c; break
    for c in CAND_RET:
        if c in df.columns: ret = c; break
    if y_pred is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) >= 1:
            y_pred = num_cols[0]
    return y_true, y_pred, date, ret

def detect_task(y):
    if y is None:
        return "unknown"
    u = pd.Series(y).dropna().unique()
    try:
        uset = set(pd.Series(u).astype(int).tolist())
        if len(uset) <= 3 and uset.issubset({0,1}):
            return "classification"
    except Exception:
        pass
    return "regression"

def equity_metrics(ret):
    ret = np.asarray(ret, dtype=float)
    eq = (1.0 + ret).cumprod()
    sharpe = 0.0 if ret.std(ddof=1)==0 else ret.mean()/ret.std(ddof=1)*np.sqrt(252)
    cagr = float(eq[-1]**(252/len(eq)) - 1.0) if len(eq)>0 else 0.0
    roll_max = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq/roll_max - 1.0)) if len(eq)>0 else 0.0
    winrate = float(np.mean(ret>0)) if len(ret)>0 else 0.0
    return {"CAGR": cagr, "Sharpe": float(sharpe), "MaxDD": float(maxdd), "WinRate": float(winrate), "FinalEquity": float(eq[-1]) if len(eq)>0 else 1.0}

def backtest_one(df, task, thr, allow_short, tc_bps):
    if "date" in df.columns:
        df = df.sort_values("date")
    if task == "classification":
        long_sig = (df["y_pred"] >= max(thr, 0.5)).astype(int)
        short_sig = (df["y_pred"] < (1 - max(thr, 0.5))).astype(int) if allow_short else 0
        signal = long_sig - short_sig
        if "ret" in df.columns:
            ret = df["ret"].astype(float).values
        else:
            if "y_true" in df.columns and set(pd.unique(df["y_true"].dropna().astype(int))).issubset({0,1}):
                ret = np.where(df["y_true"].astype(int).values>0, 1.0, -1.0)
            else:
                raise ValueError("No return column for classification backtest")
    else:
        signal = np.where(df["y_pred"] > thr, 1, np.where(df["y_pred"] < -thr, -1, 0))
        if "ret" in df.columns:
            ret = df["ret"].astype(float).values
        elif "y_true" in df.columns:
            ret = df["y_true"].astype(float).values
        else:
            raise ValueError("No numeric target/return column for regression backtest")
    pos = signal.astype(float)
    pos_prev = np.roll(pos, 1); pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    tc = turnover * (tc_bps/1e4)
    pnl = pos * ret - tc
    dfout = df.copy()
    dfout["signal"] = signal
    dfout["position"] = pos
    dfout["turnover"] = turnover
    dfout["tc"] = tc
    dfout["pnl"] = pnl
    dfout["equity"] = (1.0 + dfout["pnl"]).cumprod()
    return dfout, equity_metrics(dfout["pnl"].values)

def load_thr_map(path):
    m = {}
    if path and Path(path).exists():
        df = pd.read_csv(path)
        for _,r in df.iterrows():
            m[str(r["symbol"])] = float(r["thr"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--thr_map", default="")
    ap.add_argument("--allow_short", type=int, default=0)
    ap.add_argument("--tc_bps", type=float, default=10)
    ap.add_argument("--tag", default="preds")
    ap.add_argument("--portfolio", default="equal")
    ap.add_argument("--outdir", default="CODE/reports")
    args = ap.parse_args()

    thr_map = load_thr_map(args.thr_map)
    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    rows = []
    port_pnl = []
    symbols = []
    for f in files:
        df = pd.read_csv(f)
        yt, yp, dcol, rcol = detect_cols(df)
        if yp is None:
            continue
        if dcol is None:
            df["date"] = np.arange(len(df))
        else:
            df = df.rename(columns={dcol:"date"})
        df = df.rename(columns={yp:"y_pred"})
        if yt is not None and yt not in ["y_true"]:
            df = df.rename(columns={yt:"y_true"})
        if rcol is not None and rcol not in ["ret"]:
            df = df.rename(columns={rcol:"ret"})
        task = detect_task(df["y_true"]) if "y_true" in df.columns else "regression"
        sym = infer_symbol(f)
        sym_thr = thr_map.get(sym, args.thr)
        df_bt, m = backtest_one(df, task, sym_thr, args.allow_short, args.tc_bps)
        outdir = Path(args.outdir)/sym
        outdir.mkdir(parents=True, exist_ok=True)
        outcsv = outdir/f"backtest_{args.tag}.csv"
        df_bt.to_csv(outcsv, index=False)
        with open(outdir/f"backtest_{args.tag}_metrics.json","w") as fh:
            json.dump({"symbol": sym, "task": task, "thr": sym_thr, **m}, fh, indent=2)
        rows.append({"symbol": sym, "task": task, "thr": sym_thr, **m})
        symbols.append(sym)
        port_pnl.append(df_bt["pnl"].values)

        plt.figure()
        plt.plot(pd.to_datetime(df_bt["date"]), df_bt["equity"])
        plt.title(f"Equity: {sym}")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(outdir/f"equity_{args.tag}.png", dpi=140)
        plt.close()

    if rows:
        summ = pd.DataFrame(rows).sort_values(["Sharpe","CAGR"], ascending=[False,False])
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        summ_path = Path(args.outdir)/f"backtest_summary_{args.tag}.csv"
        summ.to_csv(summ_path, index=False)

        min_len = min(len(x) for x in port_pnl) if port_pnl else 0
        if min_len>0:
            pnl_mat = np.vstack([x[-min_len:] for x in port_pnl])
            port = pnl_mat.mean(axis=0)
            eq = (1.0 + port).cumprod()
            port_metrics = {
                "Sharpe": float(0.0 if port.std(ddof=1)==0 else port.mean()/port.std(ddof=1)*np.sqrt(252)),
                "FinalEquity": float(eq[-1]),
            }
            with open(Path(args.outdir)/f"portfolio_{args.tag}_metrics.json","w") as fh:
                json.dump(port_metrics, fh, indent=2)
            dates = np.arange(len(eq))
            plt.figure()
            plt.plot(dates, eq)
            plt.title("Portfolio Equity")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(Path(args.outdir)/f"portfolio_equity_{args.tag}.png", dpi=160)
            plt.close()

if __name__ == "__main__":
    main()
