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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds_ready/*_preds_ready.csv")
    ap.add_argument("--thr_map", required=True)
    ap.add_argument("--outdir", default="CODE/reports/auto")
    args=ap.parse_args()

    thr_map=load_thr_map(args.thr_map)
    files=sorted(Path(".").glob(args.glob))
    dfs=[]
    for f in files:
        df=pd.read_csv(f)
        if "pred" not in df.columns or "ret" not in df.columns:
            continue
        sym=infer_symbol(f)
        thr=float(thr_map.get(sym,0.0))
        if "date" in df.columns:
            date=pd.to_datetime(df["date"])
        else:
            date=pd.RangeIndex(len(df))
        pred=df["pred"].astype(float).values
        pos=np.where(pred>thr,1,np.where(pred<-thr,-1,0)).astype(float)
        pos_prev=np.roll(pos,1); pos_prev[0]=0.0
        turnover=np.abs(pos-pos_prev)
        dfo=pd.DataFrame({"date":date, f"pos_{sym}":pos, f"to_{sym}":turnover})
        dfs.append(dfo)

    if not dfs:
        return
    base=dfs[0]
    for d in dfs[1:]:
        base=base.merge(d, on="date", how="outer")
    base=base.sort_values("date").reset_index(drop=True)

    pos_cols=[c for c in base.columns if c.startswith("pos_")]
    to_cols=[c for c in base.columns if c.startswith("to_")]
    base["gross_exposure"]=np.nanmean(np.abs(base[pos_cols].values), axis=1)
    base["net_exposure"]=np.nanmean(base[pos_cols].values, axis=1)
    base["avg_turnover"]=np.nanmean(base[to_cols].values, axis=1) if to_cols else 0.0

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base.to_csv(outdir/"exposure_timeseries.csv", index=False)

    plt.figure()
    try:
        plt.plot(pd.to_datetime(base["date"]), base["gross_exposure"], label="Gross")
        plt.plot(pd.to_datetime(base["date"]), base["net_exposure"], label="Net", alpha=0.7)
    except Exception:
        plt.plot(base["gross_exposure"], label="Gross")
        plt.plot(base["net_exposure"], label="Net", alpha=0.7)
    plt.title("Portfolio Exposure (from preds)")
    plt.xlabel("Date"); plt.ylabel("Exposure")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"exposure.png", dpi=160)
    plt.close()
    print(outdir/"exposure.png")
if __name__=="__main__":
    main()
