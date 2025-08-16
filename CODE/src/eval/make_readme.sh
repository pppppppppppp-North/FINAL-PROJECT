#!/usr/bin/env bash
set -e
mkdir -p CODE/reports
{
echo "# FINAL PROJECT — Results"
echo
[ -f CODE/reports/REPORT.md ] && cat CODE/reports/REPORT.md || true
echo
[ -f CODE/reports/auto/REPORT_AUTO.md ] && cat CODE/reports/auto/REPORT_AUTO.md || true
echo
[ -f CODE/reports/auto/summary_tuned.csv ] && echo "## Tuned Summary (top 10)" && head -n 10 CODE/reports/auto/summary_tuned.csv || true
echo
[ -f CODE/reports/auto_cw/summary_cw.csv ] && echo "## Confidence-Weighted Summary (top 10)" && head -n 10 CODE/reports/auto_cw/summary_cw.csv || true
} > CODE/reports/README_COMBINED.md
echo "CODE/reports/README_COMBINED.md"
