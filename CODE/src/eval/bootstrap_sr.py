#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

def sharpe(x):
    x=np.asarray(x,dtype=float)
    s=x.std(ddof=1)
    return 0.0 if s==0 else float(x.mean()/s*np.sqrt(252))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--portfolio_csv", default="CODE/reports/auto/backtest_tuned_portfolio.csv")
    ap.add_argument("--pnl_col", default="pnl")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--out_csv", default="CODE/reports/auto/bootstrap_sr.csv")
    args=ap.parse_args()
    df=pd.read_csv(args.portfolio_csv)
    x=df[args.pnl_col].astype(float).values
    n=len(x); b=args.block
    idx=np.arange(n-b+1)
    boots=[]
    rng=np.random.default_rng(123)
    for _ in range(args.n_boot):
        picks=rng.integers(0, len(idx), size=int(np.ceil(n/b)))
        sample=np.concatenate([x[i:i+b] for i in idx[picks]])[:n]
        boots.append(sharpe(sample))
    boots=np.array(boots)
    out=pd.DataFrame({"sr":boots})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(args.out_csv)
if __name__=="__main__":
    main()
