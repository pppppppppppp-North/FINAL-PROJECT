#!/usr/bin/env python3
import argparse, re, math, json
from pathlib import Path
import numpy as np, pandas as pd

def infer_symbol(path):
    stem = Path(path).stem
    return re.sub(r"_preds_ready$","",stem)

def sharpe(ret):
    ret=np.asarray(ret,dtype=float)
    s=ret.std(ddof=1)
    return 0.0 if s==0 else float(ret.mean()/s*np.sqrt(252))

def pnl_from_pred_ret(pred, ret, thr, tc_bps):
    pred=np.asarray(pred,dtype=float); ret=np.asarray(ret,dtype=float)
    sig=np.where(pred>thr,1,np.where(pred<-thr,-1,0)).astype(float)
    sig_prev=np.roll(sig,1); sig_prev[0]=0.0
    turnover=np.abs(sig-sig_prev)
    tc=turnover*(tc_bps/1e4)
    pnl=sig*ret - tc
    return pnl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--val_frac", type=float, default=0.6)
    ap.add_argument("--tc_bps", type=float, default=10.0)
    ap.add_argument("--out_csv", default="CODE/reports/auto/best_thresholds_ready.csv")
    args=ap.parse_args()

    rows=[]
    for f in sorted(Path(".").glob(args.glob)):
        df=pd.read_csv(f)
        if not {"pred","ret"}.issubset(df.columns): 
            continue
        if "date" in df.columns:
            df=df.sort_values("date")
        n=len(df); nval=max(10,int(n*args.val_frac))
        val=df.iloc[:nval]; test=df.iloc[nval:]
        if len(test)<5 or len(val)<5: 
            continue
        abs_pred=np.abs(val["pred"].values)
        qgrid=[0.0,0.25,0.5,0.75,0.90,0.95,0.98]
        cands=sorted(set([float(np.quantile(abs_pred,q)) for q in qgrid]))
        best_thr=0.0; best_sh=-1e9
        for t in cands:
            pnl=pnl_from_pred_ret(val["pred"], val["ret"], t, args.tc_bps)
            sh=sharpe(pnl)
            if sh>best_sh:
                best_sh=sh; best_thr=t
        sym=infer_symbol(f)
        rows.append({"symbol":sym,"thr":best_thr,"val_sharpe":best_sh,"tc_bps":args.tc_bps,"n_val":len(val),"n_test":len(test)})
    if rows:
        out=pd.DataFrame(rows).sort_values("symbol")
        Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True)
        out.to_csv(args.out_csv,index=False)
        print(args.out_csv)
    else:
        print("NO_ROWS")
if __name__=="__main__":
    main()
