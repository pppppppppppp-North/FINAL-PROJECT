#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json

def load_json(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception:
        return {}

def main():
    base=Path("CODE/reports/auto")
    summ_auto=base/"summary_auto.csv"
    summ_tuned=base/"summary_tuned.csv"
    port_auto=base/"portfolio_equity_auto.png"
    port_tuned=base/"portfolio_equity_tuned.png"
    pm_auto=base/"portfolio_metrics_auto.json"
    pm_tuned=base/"portfolio_metrics_tuned.json"
    lines=[]
    lines.append("# FINAL PROJECT — Auto Backtests\n")
    if summ_auto.exists():
        df=pd.read_csv(summ_auto).sort_values(["Sharpe","FinalEquity"], ascending=[False,False])
        lines.append("## Per-Symbol (Auto)\n")
        lines.append(df.head(20).to_markdown(index=False)); lines.append("")
    if summ_tuned.exists():
        df=pd.read_csv(summ_tuned).sort_values(["Sharpe","FinalEquity"], ascending=[False,False])
        lines.append("## Per-Symbol (Tuned Thresholds)\n")
        lines.append(df.head(20).to_markdown(index=False)); lines.append("")
    if port_auto.exists():
        lines.append("### Portfolio Equity (Auto)\n")
        lines.append("![Portfolio Auto](portfolio_equity_auto.png)\n")
    if port_tuned.exists():
        lines.append("### Portfolio Equity (Tuned)\n")
        lines.append("![Portfolio Tuned](portfolio_equity_tuned.png)\n")
    out=base/"REPORT_AUTO.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
if __name__=="__main__":
    main()
