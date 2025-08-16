#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def infer_symbol(path):
    stem = Path(path).stem
    return re.sub(r"_preds_ready$","",stem)

def load_thr_map(path):
    m={}
    p=Path(path)
    if p.exists():
        df=pd.read_csv(p)
        for _,r in df.iterrows():
            m[str(r["symbol"])]=float(r["thr"])
    return m

def backtest_symbol(df, scale, cap, tc_bps):
    if "date" in df.columns:
        df=df.sort_values("date")
    pred=df["pred"].astype(float).values
    ret=df["ret"].astype(float).values
    w_raw = pred / (scale if scale>0 else (np.median(np.abs(pred))+1e-9))
    w = np.clip(w_raw, -cap, cap)
    w_prev = np.roll(w,1); w_prev[0]=0.0
    turnover = np.abs(w - w_prev)
    tc = turnover * (tc_bps/1e4)
    pnl = w * ret - tc
    out = df.copy()
    out["weight"]=w
    out["turnover"]=turnover
    out["tc"]=tc
    out["pnl"]=pnl
    out["equity"]=(1.0+out["pnl"]).cumprod()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--thr_map", default="CODE/reports/auto/best_thresholds_ready.csv")
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--tc_bps", type=float, default=10.0)
    ap.add_argument("--target_vol", type=float, default=0.0)
    ap.add_argument("--outdir", default="CODE/reports/auto_cw")
    ap.add_argument("--tag", default="cw")
    args=ap.parse_args()

    thr_map = load_thr_map(args.thr_map)
    files = sorted(Path(".").glob(args.glob))
    rows = []
    pnl_list = []
    dates_list = []

    for f in files:
        df = pd.read_csv(f)
        if not {"pred","ret"}.issubset(df.columns):
            continue
        sym = infer_symbol(f)
        scale = float(thr_map.get(sym, np.median(np.abs(pd.to_numeric(df["pred"], errors="coerce").fillna(0)))))
        bt = backtest_symbol(df, scale, args.cap, args.tc_bps)
        od = Path(args.outdir)/f"{sym}_preds_ready"
        od.mkdir(parents=True, exist_ok=True)
        bt.to_csv(od/f"backtest_{args.tag}.csv", index=False)
        plt.figure()
        x = pd.to_datetime(bt["date"]) if "date" in bt.columns else range(len(bt))
        plt.plot(x, bt["equity"])
        plt.title(f"Equity: {sym} (cap={args.cap}, scale={scale:.4g})")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(od/f"equity_{args.tag}.png", dpi=160)
        plt.close()
        ret = bt["pnl"].values
        eq = (1.0+ret).cumprod()
        s = ret.std(ddof=1)
        sh = 0.0 if s==0 else float(ret.mean()/s*np.sqrt(252))
        rows.append({"symbol":sym,"scale":scale,"Sharpe":sh,"FinalEquity":float(eq[-1])})
        pnl_list.append(ret)
        dates_list.append(bt["date"].values if "date" in bt.columns else np.arange(len(bt)))

    if rows:
        pd.DataFrame(rows).sort_values(["Sharpe","FinalEquity"], ascending=[False,False]).to_csv(Path(args.outdir)/f"summary_{args.tag}.csv", index=False)
        m = min(len(x) for x in pnl_list)
        if m>0:
            mat = np.vstack([x[-m:] for x in pnl_list])
            port = mat.mean(axis=0)
            if args.target_vol and args.target_vol>0:
                daily_target = args.target_vol/np.sqrt(252.0)
                roll = pd.Series(port).rolling(60, min_periods=20).std(ddof=1).replace(0, np.nan).fillna(method="bfill").fillna(method="ffill")
                lev = (daily_target/(roll+1e-12)).clip(upper=3.0).values
                port = port * lev
            peq = (1.0 + pd.Series(port)).cumprod().values
            try:
                dates = pd.to_datetime(dates_list[0][-m:])
            except Exception:
                dates = np.arange(m)
            plt.figure()
            plt.plot(dates, peq)
            plt.title("Portfolio Equity (confidence-weighted)")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(Path(args.outdir)/f"portfolio_equity_{args.tag}.png", dpi=180)
            plt.close()
            s = port.std(ddof=1)
            sh = 0.0 if s==0 else float(port.mean()/s*np.sqrt(252))
            json.dump({"Sharpe":sh,"FinalEquity":float(peq[-1])}, open(Path(args.outdir)/f"portfolio_metrics_{args.tag}.json","w"), indent=2)
if __name__=="__main__":
    main()
