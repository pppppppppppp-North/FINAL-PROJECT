#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def sharpe(x):
    x=np.asarray(x,dtype=float)
    s=x.std(ddof=1)
    return 0.0 if s==0 else float(x.mean()/s*np.sqrt(252))

def pnl_from(pred, ret, thr, tc_bps):
    pred=np.asarray(pred,dtype=float); ret=np.asarray(ret,dtype=float)
    sig=np.where(pred>thr,1,np.where(pred<-thr,-1,0)).astype(float)
    sig_prev=np.roll(sig,1); sig_prev[0]=0.0
    turnover=np.abs(sig-sig_prev)
    tc=turnover*(tc_bps/1e-4)/1e6 if False else turnover*(tc_bps/1e4)
    pnl=sig*ret - tc
    return pnl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--tc_bps", type=float, default=10.0)
    ap.add_argument("--outdir", default="CODE/reports/auto/threshold_curves")
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(".").glob(args.glob)):
        df=pd.read_csv(f)
        if not {"pred","ret"}.issubset(df.columns): 
            continue
        apred=np.abs(df["pred"].values)
        qs=np.linspace(0.0,0.99,20)
        thrs=sorted(set([float(np.quantile(apred,q)) for q in qs]))
        vals=[]
        for t in thrs:
            pnl=pnl_from(df["pred"].values, df["ret"].values, t, args.tc_bps)
            vals.append(sharpe(pnl))
        sym=f.stem.replace("_preds_ready","")
        plt.figure()
        plt.plot(thrs, vals)
        plt.title(f"Sharpe vs Threshold: {sym}")
        plt.xlabel("Threshold"); plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.savefig(outdir/f"{sym}_thr_curve.png", dpi=160)
        plt.close()
    print(outdir)
if __name__=="__main__":
    main()
