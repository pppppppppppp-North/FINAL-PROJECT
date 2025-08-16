#!/usr/bin/env python3
from pathlib import Path
import json, pandas as pd

def load_json(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception:
        return {}

def main():
    base = Path("CODE/reports")
    back_csv = base/"backtest_summary_preds_wf_tuned.csv"
    port_json = base/"portfolio"/"portfolio_metrics.json"
    rank_csv = base/"portfolio"/"symbol_ranking.csv"
    cost_csv = base/"cost_sweep.csv"

    lines=[]
    lines.append("# FINAL PROJECT — Evaluation & Backtest Summary\n")
    if back_csv.exists():
        df = pd.read_csv(back_csv).sort_values(["Sharpe","CAGR"], ascending=[False,False])
        lines.append("## Top 10 Symbols by Sharpe\n")
        lines.append(df.head(10).to_markdown(index=False))
        lines.append("")
    if port_json.exists():
        pm = load_json(port_json)
        lines.append("## Portfolio Metrics (equal-weight)\n")
        lines.append(f"- Sharpe: {pm.get('Sharpe',0):.3f}\n- CAGR: {pm.get('CAGR',0):.2%}\n- MaxDD: {pm.get('MaxDD',0):.2%}\n- WinRate: {pm.get('WinRate',0):.2%}\n- Final Equity: {pm.get('FinalEquity',1.0):.2f}\n")
        lines.append("")
        lines.append("![Portfolio Equity](portfolio/portfolio_equity.png)\n")
        lines.append("![Rolling Sharpe](portfolio/rolling_sharpe.png)\n")
        lines.append("![Rolling Drawdown](portfolio/rolling_drawdown.png)\n")
    if Path(rank_csv).exists():
        rk = pd.read_csv(rank_csv).head(20)
        lines.append("## Per-Symbol Ranking (Top 20)\n")
        lines.append(rk.to_markdown(index=False))
        lines.append("")
    if Path(cost_csv).exists():
        cc = pd.read_csv(cost_csv)
        lines.append("## Transaction Cost Sensitivity\n")
        lines.append(cc.to_markdown(index=False))
        lines.append("")
    out = base/"REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
if __name__ == "__main__":
    main()
