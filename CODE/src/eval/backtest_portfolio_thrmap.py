#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def infer_symbol(path):
    stem = Path(path).stem
    return re.sub(r"_preds_ready$","",stem)

def pnl_from_pred_ret(pred, ret, thr, tc_bps):
    pred=np.asarray(pred,dtype=float); ret=np.asarray(ret,dtype=float)
    sig=np.where(pred>thr,1,np.where(pred<-thr,-1,0)).astype(float)
    sig_prev=np.roll(sig,1); sig_prev[0]=0.0
    turnover=np.abs(sig-sig_prev)
    tc=turnover*(tc_bps/1e4)
    pnl=sig*ret - tc
    eq=(1.0+pnl).cumprod()
    s=pnl.std(ddof=1)
    sh=0.0 if s==0 else float(pnl.mean()/s*np.sqrt(252))
    return pnl, eq, sh

def load_thr_map(path):
    m={}
    if path and Path(path).exists():
        df=pd.read_csv(path)
        for _,r in df.iterrows():
            m[str(r["symbol"])]=float(r["thr"])
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--thr_map", required=True)
    ap.add_argument("--tc_bps", type=float, default=10.0)
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args=ap.parse_args()

    thr_map=load_thr_map(args.thr_map)
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; pnl_list=[]; dates_list=[]
    for f in sorted(Path(".").glob(args.glob)):
        df=pd.read_csv(f)
        if not {"pred","ret"}.issubset(df.columns): 
            continue
        sym=infer_symbol(f)
        thr=float(thr_map.get(sym,0.0))
        pnl, eq, sh = pnl_from_pred_ret(df["pred"].values, df["ret"].values, thr, args.tc_bps)
        dfx=df.copy()
        if "date" in dfx.columns:
            dfx=dfx.sort_values("date")
        dfx["pnl"]=pnl; dfx["equity"]=eq
        od=outdir/f"{sym}_preds_ready"; od.mkdir(parents=True,exist_ok=True)
        dfx.to_csv(od/f"backtest_{args.tag}.csv", index=False)
        plt.figure()
        x = pd.to_datetime(dfx["date"]) if "date" in dfx.columns else range(len(dfx))
        plt.plot(x, dfx["equity"])
        plt.title(f"Equity: {sym} (thr={thr:.4f})")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(od/f"equity_{args.tag}.png", dpi=160)
        plt.close()
        rows.append({"symbol":sym,"thr":thr,"Sharpe":sh,"FinalEquity":float(eq[-1])})
        pnl_list.append(pnl)
        dates_list.append(dfx["date"].values if "date" in dfx.columns else np.arange(len(dfx)))
    if rows:
        pd.DataFrame(rows).sort_values(["Sharpe","FinalEquity"], ascending=[False,False]).to_csv(outdir/f"summary_{args.tag}.csv", index=False)
        m=min(len(x) for x in pnl_list)
        if m>0:
            mat=np.vstack([x[-m:] for x in pnl_list])
            port=mat.mean(axis=0)
            peq=(1.0+pd.Series(port)).cumprod().values
            try:
                dates=pd.to_datetime(dates_list[0][-m:])
            except Exception:
                dates=np.arange(m)
            plt.figure()
            plt.plot(dates, peq)
            plt.title("Portfolio Equity (tuned)")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(outdir/f"portfolio_equity_{args.tag}.png", dpi=180)
            plt.close()
            s=port.std(ddof=1); sh=0.0 if s==0 else float(port.mean()/s*np.sqrt(252))
            json.dump({"Sharpe":sh,"FinalEquity":float(peq[-1])}, open(outdir/f"portfolio_metrics_{args.tag}.json","w"), indent=2)
if __name__=="__main__":
    main()
