#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd

CAND_TRUE = ["y_true","true","label","target","y","actual","direction"]
CAND_PRED = ["y_pred","pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]
CAND_RET  = ["ret","return","returns","y_ret","next_ret","r","y_next"]
CAND_DATE = ["date","timestamp","time","ds"]

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def detect_cols(df):
    yt=yp=rt=dt=None
    for c in CAND_TRUE:
        if c in df.columns: yt=c; break
    for c in CAND_PRED:
        if c in df.columns: yp=c; break
    for c in CAND_RET:
        if c in df.columns: rt=c; break
    for c in CAND_DATE:
        if c in df.columns: dt=c; break
    if yp is None:
        num_cols=[c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if num_cols: yp=num_cols[0]
    return yt,yp,rt,dt

def detect_task(y):
    if y is None: return "regression"
    u = pd.Series(y).dropna().unique()
    try:
        s=set(pd.Series(u).astype(int).tolist())
        if len(s)<=3 and s.issubset({0,1}): return "classification"
    except Exception:
        pass
    return "regression"

def equity_metrics(ret):
    ret=np.asarray(ret,dtype=float)
    if ret.size==0: return {"Sharpe":0.0,"FinalEquity":1.0}
    eq=(1.0+ret).cumprod()
    sharpe=0.0 if ret.std(ddof=1)==0 else ret.mean()/ret.std(ddof=1)*np.sqrt(252)
    return {"Sharpe":float(sharpe),"FinalEquity":float(eq[-1])}

def backtest_vec(y_pred, y_true, ret, task, thr, allow_short, tc_bps):
    if task=="classification":
        t=max(thr,0.5)
        long_sig=(y_pred>=t).astype(int)
        short_sig=(y_pred<(1-t)).astype(int) if allow_short else 0
        signal=long_sig-short_sig
        if ret is None:
            ret=np.where((y_true.astype(int)>0),1.0,-1.0)
        else:
            ret=ret.astype(float)
    else:
        signal=np.where(y_pred>thr,1,np.where(y_pred<-thr,-1,0))
        ret=(ret if ret is not None else y_true).astype(float)
    pos=signal.astype(float)
    pos_prev=np.roll(pos,1); pos_prev[0]=0.0
    turnover=np.abs(pos-pos_prev)
    tc=turnover*(tc_bps/1e4)
    pnl=pos*ret - tc
    return pnl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob",default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--tc_bps",type=float,default=10.0)
    ap.add_argument("--allow_short",type=int,default=0)
    ap.add_argument("--out_csv",default="CODE/reports/best_thresholds_preds.csv")
    args=ap.parse_args()

    files=sorted(Path(".").glob(args.glob))
    rows=[]
    for f in files:
        df=pd.read_csv(f)
        yt,yp,rt,dt=detect_cols(df)
        if yp is None: continue
        sym=infer_symbol(f)
        y_pred=df[yp].values.astype(float)
        y_true=df[yt].values if yt is not None else None
        ret=df[rt].values if rt is not None else None
        task=detect_task(y_true)

        if task=="classification":
            thrs=np.round(np.linspace(0.50,0.90,9),2)
            best=(None,-1e9)
            for t in thrs:
                pnl=backtest_vec(y_pred,y_true,ret,task,t,args.allow_short,args.tc_bps)
                m=equity_metrics(pnl)
                if m["Sharpe"]>best[1]:
                    best=(t,m["Sharpe"])
            rows.append({"symbol":sym,"task":task,"thr":best[0],"allow_short":args.allow_short,"tc_bps":args.tc_bps,"best_sharpe":best[1]})
        else:
            qgrid=[0.0,0.25,0.5,0.75,0.90,0.95,0.98]
            abs_pred=np.abs(y_pred)
            cand=[float(np.quantile(abs_pred,q)) for q in qgrid]
            cand=sorted(set([round(c,6) for c in cand]))
            best=(0.0,-1e9)
            for t in cand:
                pnl=backtest_vec(y_pred,y_true,ret,task,t,args.allow_short,args.tc_bps)
                m=equity_metrics(pnl)
                if m["Sharpe"]>best[1]:
                    best=(t,m["Sharpe"])
            rows.append({"symbol":sym,"task":task,"thr":best[0],"allow_short":args.allow_short,"tc_bps":args.tc_bps,"best_sharpe":best[1]})
    if rows:
        out=pd.DataFrame(rows).sort_values(["task","symbol"])
        Path(args.out_csv).parent.mkdir(parents=True,exist_ok=True)
        out.to_csv(args.out_csv,index=False)
        print(args.out_csv)
    else:
        print("NO_ROWS")
if __name__=="__main__":
    main()
