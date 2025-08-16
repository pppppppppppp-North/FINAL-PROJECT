#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def load_summary(path):
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--tuned_csv", default="CODE/reports/best_thresholds_preds.csv")
    ap.add_argument("--tc_list", default="0,5,10,20,30,50")
    ap.add_argument("--out_csv", default="CODE/reports/cost_sweep.csv")
    args = ap.parse_args()

    thrs = pd.read_csv(args.tuned_csv).set_index("symbol")["thr"].to_dict()
    tc_vals = [float(x) for x in args.tc_list.split(",")]

    rows=[]
    for tc in tc_vals:
        outdir = Path("CODE/reports/tmp_cost"); outdir.mkdir(parents=True, exist_ok=True)
        cmd = f'python3 -m CODE.src.eval.backtest_from_preds --glob "{args.glob}" --tc_bps {tc} --tag preds_tc_{int(tc)} --thr_map "{args.tuned_csv}"'
        rc = os.system(cmd)
        if rc != 0: continue
        summ = pd.read_csv(f"CODE/reports/backtest_summary_preds_tc_{int(tc)}.csv")
        sharpe_mean = float(summ["Sharpe"].mean())
        cagr_mean = float(summ["CAGR"].mean())
        rows.append({"tc_bps":tc,"mean_sharpe":sharpe_mean,"mean_cagr":cagr_mean})
    if rows:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(args.out_csv)
    else:
        print("NO_ROWS")

if __name__ == "__main__":
    import os
    main()
