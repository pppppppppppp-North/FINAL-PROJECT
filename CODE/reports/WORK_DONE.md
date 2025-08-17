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


---

## 📊 Numerical Results

### Evaluation Summary


| symbol    | task           |   accuracy |   precision |   recall |       f1 |      auc |
|:----------|:---------------|-----------:|------------:|---------:|---------:|---------:|
| ADVANC.BK | classification |   0.607407 |    0.634146 | 0.40625  | 0.495238 | 0.597491 |
| AOT.BK    | classification |   0.52963  |    0.357143 | 0.130435 | 0.191083 | 0.478121 |
| AWC.BK    | classification |   0.52963  |    0.449102 | 0.432277 | 0.440529 | 0.517434 |
| BANPU.BK  | classification |   0.517284 |    0.423077 | 0.459701 | 0.440629 | 0.508798 |
| BBL.BK    | classification |   0.548148 |    0.51632  | 0.461538 | 0.487395 | 0.542548 |
| BCP.BK    | classification |   0.57284  |    0.604651 | 0.209115 | 0.310757 | 0.546205 |
| BDMS.BK   | classification |   0.58642  |    0.535211 | 0.111765 | 0.184915 | 0.520776 |
| BEM.BK    | classification |   0.562963 |    0.377193 | 0.131902 | 0.195455 | 0.492604 |
| BH.BK     | classification |   0.591358 |    0.6      | 0.479695 | 0.533145 | 0.588405 |
| BJC.BK    | classification |   0.579012 |    0.560606 | 0.106017 | 0.178313 | 0.521555 |



### Auto Backtest (Top 10)


| symbol                |    Sharpe |   FinalEquity |
|:----------------------|----------:|--------------:|
| ADVANC.BK_preds_ready |  2.36791  |            -0 |
| AOT.BK_preds_ready    | -1.80048  |             0 |
| AWC.BK_preds_ready    | -1.03979  |            -0 |
| BANPU.BK_preds_ready  | -1.64715  |             0 |
| BBL.BK_preds_ready    |  0.332323 |            -0 |
| BCP.BK_preds_ready    |  1.32859  |             0 |
| BDMS.BK_preds_ready   |  0.327743 |            -0 |
| BEM.BK_preds_ready    | -1.469    |             0 |
| BH.BK_preds_ready     |  1.99314  |             0 |
| BJC.BK_preds_ready    |  0.547047 |             0 |



### Confidence-Weighted Backtest (Top 10)


| symbol    |   scale |   Sharpe |   FinalEquity |
|:----------|--------:|---------:|--------------:|
| ADVANC.BK |       0 |  2.36791 |         -0    |
| DELTA.BK  |       0 |  2.11989 |          0    |
| BH.BK     |       0 |  1.99314 |          0    |
| WHA.BK    |       0 |  1.82356 |          0    |
| KBANK.BK  |       0 |  1.79726 |          0    |
| TRUE.BK   |       0 |  1.64249 |         -0    |
| TOP.BK    |       0 |  1.63391 |         -0    |
| OR.BK     |       0 |  1.58389 |        254.85 |
| IVL.BK    |       0 |  1.45842 |         -0    |
| CBG.BK    |       0 |  1.41269 |         -0    |



### Walk-Forward Backtest (Top 10)


| symbol    |   Sharpe_WF |   FinalEquity_WF |
|:----------|------------:|-----------------:|
| ADVANC.BK |    3.05198  |                0 |
| DELTA.BK  |    2.32155  |               -0 |
| WHA.BK    |    1.78242  |               -0 |
| TOP.BK    |    1.48833  |               -0 |
| TRUE.BK   |    1.27732  |                0 |
| KBANK.BK  |    1.1007   |                0 |
| TCAP.BK   |    0.957729 |               -0 |
| IVL.BK    |    0.929705 |               -0 |
| CBG.BK    |    0.907603 |               -0 |
| CPF.BK    |    0.822777 |               -0 |



### Bootstrap Sharpe (first 10 samples)


|        sr |
|----------:|
|  2.08161  |
| -0.216752 |
|  2.95779  |
|  1.49139  |
|  1.57329  |
|  1.80934  |
|  3.47554  |
|  2.21303  |
|  3.508    |
|  3.92968  |


Mean SR: 2.036, Std SR: 0.924