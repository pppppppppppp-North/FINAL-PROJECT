#!/usr/bin/env python3
import argparse, re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def infer_symbol(p):
    stem = Path(p).stem
    return re.sub(r"_preds_ready$","",stem)

def sharpe(x):
    x=np.asarray(x,dtype=float)
    s=x.std(ddof=1)
    return 0.0 if s==0 else float(x.mean()/s*np.sqrt(252))

def pnl_from(pred, ret, thr, tc_bps):
    pred=np.asarray(pred,dtype=float); ret=np.asarray(ret,dtype=float)
    sig=np.where(pred>thr,1,np.where(pred<-thr,-1,0)).astype(float)
    sig_prev=np.roll(sig,1); sig_prev[0]=0.0
    turnover=np.abs(sig-sig_prev)
    tc=turnover*(tc_bps/1e4)
    pnl=sig*ret - tc
    return pnl

def tune_thr(pred, ret, tc_bps):
    ap=np.abs(pred.astype(float).values)
    qs=np.linspace(0.0,0.98,15)
    thrs=sorted(set([float(np.quantile(ap,q)) for q in qs]))
    best=0.0; best_sh=-1e9
    for t in thrs:
        pnl=pnl_from(pred.values, ret.values, t, tc_bps)
        sh=sharpe(pnl)
        if sh>best_sh: best_sh, best = sh, t
    return best

def walkforward_one(df, n_init=252, step=21, tc_bps=10):
    df=df.dropna(subset=["pred","ret"]).copy()
    if "date" in df.columns:
        try:
            df["date"]=pd.to_datetime(df["date"])
            df=df.sort_values("date")
        except Exception:
            df=df.reset_index(drop=True)
    else:
        df=df.reset_index(drop=True)
    n=len(df)
    pnl=np.zeros(n)
    thrs=np.full(n, 0.0)
    i=n_init
    if i>=n: 
        pnl[:]=0.0
        return pnl, thrs
    while i<n:
        tr_end=i
        te_end=min(i+step, n)
        thr=tune_thr(df.loc[:tr_end-1,"pred"], df.loc[:tr_end-1,"ret"], tc_bps)
        pp=pnl_from(df.loc[tr_end:te_end-1,"pred"].values, df.loc[tr_end:te_end-1,"ret"].values, thr, tc_bps)
        pnl[tr_end:te_end]=pp
        thrs[tr_end:te_end]=thr
        i=te_end
    return pnl, thrs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--n_init", type=int, default=252)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--tc_bps", type=float, default=10.0)
    ap.add_argument("--outdir", default="CODE/reports/walkforward")
    args=ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows=[]; pnl_list=[]; dates_list=[]
    for f in sorted(Path(".").glob(args.glob)):
        df=pd.read_csv(f)
        if not {"pred","ret"}.issubset(df.columns): 
            continue
        sym=infer_symbol(f)
        pnl, thrs = walkforward_one(df, n_init=args.n_init, step=args.step, tc_bps=args.tc_bps)
        dfx=df.copy()
        dfx["pnl_wf"]=pnl
        dfx["thr_wf"]=thrs
        dfx["equity_wf"]=(1.0+pd.Series(pnl)).cumprod().values
        od=Path(args.outdir)/f"{sym}_preds_ready"
        od.mkdir(parents=True, exist_ok=True)
        dfx.to_csv(od/"walkforward.csv", index=False)

        import matplotlib.pyplot as plt
        plt.figure()
        x = pd.to_datetime(dfx["date"]) if "date" in dfx.columns else range(len(dfx))
        plt.plot(x, dfx["equity_wf"])
        plt.title(f"Walk-Forward Equity: {sym}")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(od/"equity_wf.png", dpi=160)
        plt.close()

        sh=sharpe(pnl)
        rows.append({"symbol":sym,"Sharpe_WF":sh,"FinalEquity_WF":float(dfx["equity_wf"].iloc[-1])})
        pnl_list.append(pnl)
        dates_list.append(dfx["date"].values if "date" in dfx.columns else np.arange(len(dfx)))

    if rows:
        pd.DataFrame(rows).sort_values(["Sharpe_WF","FinalEquity_WF"], ascending=[False,False]).to_csv(Path(args.outdir)/"summary_wf.csv", index=False)
        m=min(len(x) for x in pnl_list) if pnl_list else 0
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
            plt.title("Portfolio Equity — Walk-Forward")
            plt.xlabel("Date"); plt.ylabel("Equity")
            plt.tight_layout()
            plt.savefig(Path(args.outdir)/"portfolio_equity_wf.png", dpi=180)
            plt.close()
            s=port.std(ddof=1); sh=0.0 if s==0 else float(port.mean()/s*np.sqrt(252))
            json.dump({"Sharpe_WF":sh,"FinalEquity_WF":float(peq[-1])}, open(Path(args.outdir)/"portfolio_metrics_wf.json","w"), indent=2)
if __name__=="__main__":
    main()
