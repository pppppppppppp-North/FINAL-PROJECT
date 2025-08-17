# FINAL PROJECT — Work Log (Aug 2025)

This document summarizes all progress achieved so far.

---

## 1. Setup
- Project root organized under `FINAL PROJECT/CODE`.
- Connected to **OneDrive** for local syncing and **GitHub** (`pppppppppppp-North/FINAL-PROJECT`) for version control.
- Configured `.gitattributes` with **Git LFS** for large files (models, images, datasets).
- Added `.gitignore`, `requirements.txt`, README, LICENSE.
- GitHub Actions **CI workflow** created to check imports and upload artifacts.

---

## 2. Data Pipeline
- ✅ Collected **raw SET50 stock data**.
- ✅ Generated **technical indicators**.
- ✅ Converted into **windowed sequences** ready for training.

---

## 3. Model Training
- Implemented and tested:
  - `bilstm_baseline.py` (single model training)
  - `train_all.py` (batch trainer across multiple stocks)
- Trained models saved under `CODE/reports/`.
- Produced **training histories** and accuracy logs.

---

## 4. Evaluation
- Ran `evaluate_from_preds` on `*_test_preds.csv` files.
- Created **per-stock evaluation JSON** and a combined **summary CSV**.
- Checked and fixed missing column issues (`y_true` vs `ret`).

---

## 5. Diagnostics
- Generated **learning curves** (`*_learning_curves.png`).
- Inspected per-stock diagnostic plots (confusion, residuals, distributions).
- Some plots looked identical → possible plotting bug flagged.

---

## 6. Backtesting
- Built scripts for **per-stock backtesting** (`backtest_from_preds_explicit.py`).
- Ran **portfolio backtesting** (`backtest_portfolio.py`) with transaction costs.
- Produced:
  - Equity curves per stock
  - Portfolio equity curve (`portfolio_equity_auto.png`)
  - Portfolio summary metrics (Sharpe, final equity)

---

## 7. Threshold & Exposure Analysis
- Tuned **per-stock thresholds** for signal generation.
- Generated **threshold curves** (`*_thr_curve.png`).
- Built **exposure summary**:
  - Gross exposure
  - Net exposure
  - Turnover
- Exported exposure time-series + plots.

---

## 8. Confidence-Weighted Strategy
- Implemented **confidence-weighted backtesting**:
  - Position size proportional to prediction magnitude.
  - Capped leverage, added transaction costs.
- Produced **per-stock** and **portfolio** confidence-weighted equity curves.

---

## 9. Walk-Forward Validation
- Added `walkforward_ready.py`:
  - Rolling threshold tuning on training windows
  - Applied forward on test windows
- Exported **walk-forward equity per stock** and **portfolio equity_wf.png**.

---

## 10. Packaging & Final Results
- Bundled everything into `CODE/reports/FINAL_RESULTS.tar.gz`.
- Prepared combined markdown reports:
  - `REPORT.md`
  - `auto/REPORT_AUTO.md`
  - `README_COMBINED.md`

---

## 11. GitHub Integration
- Full repository uploaded:
  - Code
  - Reports
  - Artifacts
- CI/CD pipeline enabled (imports + report artifacts).
- Tagged release: `v0.1`.

---

## ✅ Summary
In just two days we:
- Built a complete **data → model → evaluation → backtest → portfolio → report** pipeline.
- Fixed column mismatches, directory issues, and automated report generation.
- Synced all code + artifacts to GitHub.
- Added robustness (confidence-weighted, walk-forward, bootstrap SR).
- Packaged everything reproducibly.

Next step: start a **new clean folder** (fresh repo or branch), and stretch this workflow over the full semester with more thorough analysis & refinement.
