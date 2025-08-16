#!/usr/bin/env python3
import re, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DATE_CANDS = ["date","ds","timestamp","time"]
PRED_CANDS = ["pred","y_pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def pick_date_col(df):
    for c in DATE_CANDS:
        if c in df.columns: return c
    return None

def pick_pred_col(df):
    for c in PRED_CANDS:
        if c in df.columns: return c
    num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    num = [c for c in num if c.lower() not in {"ret","return","returns"}]
    if not num: return None
    stds = {c: float(df[c].astype(float).std()) for c in num}
    return max(stds, key=stds.get)

def backtest(df, task, thr, allow_short, tc_bps):
    if "date" in df.columns:
        df = df.sort_values("date")
    if task == "classification":
        t = max(thr, 0.5)
        long_sig = (df["y_pred"] >= t).astype(int)
        short_sig = (df["y_pred"] < (1-t)).astype(int) if allow_short else 0
        signal = long_sig - short_sig
        if "ret" in df.columns:
            ret = df["ret"].astype(float).values
        elif "y_true" in df.columns:
            ret = np.where(df["y_true"].astype(int)>0, 1.0, -1.0)
        else:
            raise ValueError("need ret or y_true")
    else:
        signal = np.where(df["y_pred"] > thr, 1, np.where(df["y_pred"] < -thr, -1, 0))
        ret = (df["ret"] if "ret" in df.columns else df["y_true"]).astype(float).values
    pos = signal.astype(float)
    pos_prev = np.roll(pos, 1); pos_prev[0] = 0.0
    turnover = np.abs(pos - pos_prev)
    tc = turnover * (tc_bps/1e4)
    pnl = pos * ret - tc
    df["signal"]=signal; df["position"]=pos; df["turnover"]=turnover; df["tc"]=tc; df["pnl"]=pnl
    df["equity"] = (1.0 + df["pnl"]).cumprod()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--tc_bps", type=float, default=10)
    ap.add_argument("--thr_map", default="CODE/reports/best_thresholds_preds.csv")
    ap.add_argument("--allow_short", type=int, default=0)
    ap.add_argument("--tag", default="auto")
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args = ap.parse_args()

    thr_map = {}
    p = Path(args.thr_map)
    if p.exists():
        thr_map = pd.read_csv(p).set_index("symbol")["thr"].to_dict()

    files = sorted(Path(".").glob(args.glob))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        sym = infer_symbol(f)
        dcol = pick_date_col(df)
        if dcol: df = df.rename(columns={dcol:"date"})
        if "ret" not in df.columns: 
            continue
        pcol = pick_pred_col(df)
        if pcol is None: 
            continue
        df = df.rename(columns={pcol:"y_pred"})
        if "y_true" in df.columns:
            try:
                u = set(df["y_true"].dropna().astype(int).unique().tolist())
                task = "classification" if u.issubset({0,1}) else "regression"
            except Exception:
                task = "regression"
        else:
            task = "regression"
        thr = float(thr_map.get(sym, 0.0))
        bt = backtest(df.copy(), task, thr, args.allow_short, args.tc_bps)

        outdir = Path(args.outdir)/sym
        outdir.mkdir(parents=True, exist_ok=True)
        bt.to_csv(outdir/f"backtest_{args.tag}.csv", index=False)
        plt.figure()
        x = pd.to_datetime(bt["date"]) if "date" in bt.columns else range(len(bt))
        plt.plot(x, bt["equity"])
        plt.title(f"Equity: {sym}")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(outdir/f"equity_{args.tag}.png", dpi=160)
        plt.close()

        ret = bt["pnl"].values
        eq = (1.0 + ret).cumprod()
        sharpe = 0.0 if ret.std(ddof=1)==0 else ret.mean()/ret.std(ddof=1)*np.sqrt(252)
        rows.append({"symbol":sym,"task":task,"thr":thr,"Sharpe":float(sharpe),"FinalEquity":float(eq[-1])})

        with open(outdir/f"meta_{args.tag}.json","w") as fh:
            json.dump({"symbol":sym,"date_col":(dcol or "index"),"pred_col":pcol,"task":task,"thr":thr,"tc_bps":args.tc_bps}, fh, indent=2)

    if rows:
        pd.DataFrame(rows).sort_values("symbol").to_csv(Path(args.outdir)/f"summary_{args.tag}.csv", index=False)

        # portfolio
        eqs=[]
        for f in sorted(Path(args.outdir).glob("*/backtest_*.csv")):
            d = pd.read_csv(f)
            eqs.append(d["pnl"].values)
        m = min(len(x) for x in eqs) if eqs else 0
        if m>0:
            mat = np.vstack([x[-m:] for x in eqs])
            port = mat.mean(axis=0)
            peq = (1.0 + pd.Series(port)).cumprod().values
            dates = pd.read_csv(sorted(Path(args.outdir).glob("*/backtest_*.csv"))[0])["date"].values[-m:]
            plt.figure()
            plt.plot(pd.to_datetime(dates), peq)
            plt.title("Portfolio Equity (auto)")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(Path(args.outdir)/"portfolio_equity_auto.png", dpi=180)
            plt.close()

if __name__ == "__main__":
    main()
