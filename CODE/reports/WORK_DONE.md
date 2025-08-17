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

## 📊 Numerical Results (All 50 Stocks)

### Evaluation Summary


| symbol    | task           |   accuracy |   precision |     recall |        f1 |      auc |
|:----------|:---------------|-----------:|------------:|-----------:|----------:|---------:|
| ADVANC.BK | classification |   0.607407 |    0.634146 | 0.40625    | 0.495238  | 0.597491 |
| AOT.BK    | classification |   0.52963  |    0.357143 | 0.130435   | 0.191083  | 0.478121 |
| AWC.BK    | classification |   0.52963  |    0.449102 | 0.432277   | 0.440529  | 0.517434 |
| BANPU.BK  | classification |   0.517284 |    0.423077 | 0.459701   | 0.440629  | 0.508798 |
| BBL.BK    | classification |   0.548148 |    0.51632  | 0.461538   | 0.487395  | 0.542548 |
| BCP.BK    | classification |   0.57284  |    0.604651 | 0.209115   | 0.310757  | 0.546205 |
| BDMS.BK   | classification |   0.58642  |    0.535211 | 0.111765   | 0.184915  | 0.520776 |
| BEM.BK    | classification |   0.562963 |    0.377193 | 0.131902   | 0.195455  | 0.492604 |
| BH.BK     | classification |   0.591358 |    0.6      | 0.479695   | 0.533145  | 0.588405 |
| BJC.BK    | classification |   0.579012 |    0.560606 | 0.106017   | 0.178313  | 0.521555 |
| BTS.BK    | classification |   0.609877 |    0.556522 | 0.194529   | 0.288288  | 0.54425  |
| CBG.BK    | classification |   0.598765 |    0.568116 | 0.526882   | 0.546722  | 0.59335  |
| CCET.BK   | classification |   0.555556 |    0.525316 | 0.225543   | 0.315589  | 0.52793  |
| COM7.BK   | classification |   0.511111 |    0.464481 | 0.714286   | 0.562914  | 0.53264  |
| CPALL.BK  | classification |   0.553086 |    0.5      | 0.129834   | 0.20614   | 0.512462 |
| CPF.BK    | classification |   0.603704 |    0.593496 | 0.212209   | 0.312634  | 0.552457 |
| CPN.BK    | classification |   0.541975 |    0.467249 | 0.300562   | 0.365812  | 0.51592  |
| CRC.BK    | classification |   0.545679 |    0.49505  | 0.273224   | 0.352113  | 0.521747 |
| DELTA.BK  | classification |   0.498765 |    0.661765 | 0.2        | 0.307167  | 0.536111 |
| EGCO.BK   | classification |   0.498765 |    0.403101 | 0.471299   | 0.43454   | 0.494522 |
| GPSC.BK   | classification |   0.503704 |    0.461712 | 0.557065   | 0.504926  | 0.508171 |
| HMPRO.BK  | classification |   0.546914 |    0.437055 | 0.585987   | 0.50068   | 0.554082 |
| IVL.BK    | classification |   0.624691 |    0.606667 | 0.270833   | 0.374486  | 0.57318  |
| KBANK.BK  | classification |   0.560494 |    0.59707  | 0.398533   | 0.478006  | 0.562109 |
| KKP.BK    | classification |   0.571605 |    0.511628 | 0.250712   | 0.33652   | 0.533853 |
| KTB.BK    | classification |   0.538272 |    0.542274 | 0.461538   | 0.49866   | 0.537895 |
| KTC.BK    | classification |   0.560494 |    0.555556 | 0.10989    | 0.183486  | 0.519071 |
| LH.BK     | classification |   0.593827 |    0.497449 | 0.59633    | 0.54242   | 0.594231 |
| MINT.BK   | classification |   0.561728 |    0.487032 | 0.488439   | 0.487734  | 0.552409 |
| MTC.BK    | classification |   0.503704 |    0.461538 | 0.7        | 0.556291  | 0.523333 |
| OR.BK     | classification |   0.606173 |    1        | 0.0244648  | 0.0477612 | 0.512232 |
| OSP.BK    | classification |   0.481481 |    0.452349 | 0.965616   | 0.616088  | 0.540292 |
| PTT.BK    | classification |   0.507407 |    0.448382 | 0.876506   | 0.593272  | 0.563776 |
| PTTEP.BK  | classification |   0.541975 |    0.542857 | 0.246114   | 0.338681  | 0.528717 |
| PTTGC.BK  | classification |   0.501235 |    0.467572 | 0.858726   | 0.605469  | 0.536267 |
| RATCH.BK  | classification |   0.587654 |    0.391304 | 0.0273556  | 0.0511364 | 0.499125 |
| SCC.BK    | classification |   0.519753 |    0.426471 | 0.636364   | 0.510692  | 0.540178 |
| SCGP.BK   | classification |   0.428395 |    0.400922 | 0.781437   | 0.529949  | 0.481055 |
| TCAP.BK   | classification |   0.553086 |    0.605634 | 0.114058   | 0.191964  | 0.524697 |
| TISCO.BK  | classification |   0.546914 |    0.543767 | 0.5125     | 0.527671  | 0.546494 |
| TOP.BK    | classification |   0.579012 |    0.578947 | 0.501266   | 0.537313  | 0.577139 |
| TRUE.BK   | classification |   0.555556 |    0.616352 | 0.246851   | 0.352518  | 0.549576 |
| TTB.BK    | classification |   0.539506 |    0.666667 | 0.00534759 | 0.0106101 | 0.501527 |
| TU.BK     | classification |   0.516049 |    0.433134 | 0.667692   | 0.525424  | 0.541063 |
| VGI.BK    | classification |   0.590123 |    0.522523 | 0.339181   | 0.411348  | 0.556343 |
| WHA.BK    | classification |   0.57284  |    0.605932 | 0.361111   | 0.452532  | 0.568237 |



### Auto Backtest (All Stocks)


| symbol                |      Sharpe |   FinalEquity |
|:----------------------|------------:|--------------:|
| ADVANC.BK_preds_ready |  2.36791    |   -0          |
| AOT.BK_preds_ready    | -1.80048    |    0          |
| AWC.BK_preds_ready    | -1.03979    |   -0          |
| BANPU.BK_preds_ready  | -1.64715    |    0          |
| BBL.BK_preds_ready    |  0.332323   |   -0          |
| BCP.BK_preds_ready    |  1.32859    |    0          |
| BDMS.BK_preds_ready   |  0.327743   |   -0          |
| BEM.BK_preds_ready    | -1.469      |    0          |
| BH.BK_preds_ready     |  1.99314    |    0          |
| BJC.BK_preds_ready    |  0.547047   |    0          |
| BTS.BK_preds_ready    |  0.673123   |    0          |
| CBG.BK_preds_ready    |  1.41269    |   -0          |
| CCET.BK_preds_ready   |  0.352399   |   -0          |
| COM7.BK_preds_ready   | -0.931015   |   -0          |
| CPALL.BK_preds_ready  | -0.00258739 |   -0          |
| CPF.BK_preds_ready    |  1.157      |   -0          |
| CPN.BK_preds_ready    | -0.55546    |   -0          |
| CRC.BK_preds_ready    | -0.0796588  |   -0          |
| DELTA.BK_preds_ready  |  2.11989    |    0          |
| EGCO.BK_preds_ready   | -2.1476     |   -0          |
| GPSC.BK_preds_ready   | -0.90325    |    0          |
| HMPRO.BK_preds_ready  | -1.44839    |   -0          |
| IVL.BK_preds_ready    |  1.45842    |   -0          |
| KBANK.BK_preds_ready  |  1.79726    |    0          |
| KKP.BK_preds_ready    |  0.166412   |   -0          |
| KTB.BK_preds_ready    |  0.870558   |    0          |
| KTC.BK_preds_ready    |  0.524419   |    0          |
| LH.BK_preds_ready     | -0.0576036  |    0          |
| MINT.BK_preds_ready   | -0.271986   |    0          |
| MTC.BK_preds_ready    | -1.00614    |    0          |
| OR.BK_preds_ready     |  1.58389    |  254.85       |
| OSP.BK_preds_ready    | -1.45649    |   -0          |
| PTT.BK_preds_ready    | -1.47322    |    0          |
| PTTEP.BK_preds_ready  |  0.629878   |    0          |
| PTTGC.BK_preds_ready  | -0.933503   |    0          |
| RATCH.BK_preds_ready  | -0.585484   |   -0          |
| SCC.BK_preds_ready    | -1.80176    |    0          |
| SCGP.BK_preds_ready   | -2.86439    |    0          |
| TCAP.BK_preds_ready   |  0.991513   |   -0          |
| TISCO.BK_preds_ready  |  0.946146   |    0          |
| TOP.BK_preds_ready    |  1.63391    |   -0          |
| TRUE.BK_preds_ready   |  1.64249    |   -0          |
| TTB.BK_preds_ready    |  0.320072   |   -0.00398402 |
| TU.BK_preds_ready     | -1.68027    |    0          |
| VGI.BK_preds_ready    |  0.371077   |    0          |
| WHA.BK_preds_ready    |  1.82356    |    0          |



### Confidence-Weighted Backtest (All Stocks)


| symbol    |   scale |      Sharpe |   FinalEquity |
|:----------|--------:|------------:|--------------:|
| ADVANC.BK |       0 |  2.36791    |   -0          |
| DELTA.BK  |       0 |  2.11989    |    0          |
| BH.BK     |       0 |  1.99314    |    0          |
| WHA.BK    |       0 |  1.82356    |    0          |
| KBANK.BK  |       0 |  1.79726    |    0          |
| TRUE.BK   |       0 |  1.64249    |   -0          |
| TOP.BK    |       0 |  1.63391    |   -0          |
| OR.BK     |       0 |  1.58389    |  254.85       |
| IVL.BK    |       0 |  1.45842    |   -0          |
| CBG.BK    |       0 |  1.41269    |   -0          |
| BCP.BK    |       0 |  1.32859    |    0          |
| CPF.BK    |       0 |  1.157      |   -0          |
| TCAP.BK   |       0 |  0.991513   |   -0          |
| TISCO.BK  |       0 |  0.946146   |    0          |
| KTB.BK    |       0 |  0.870558   |    0          |
| BTS.BK    |       0 |  0.673123   |    0          |
| PTTEP.BK  |       0 |  0.629878   |    0          |
| BJC.BK    |       0 |  0.547047   |    0          |
| KTC.BK    |       0 |  0.524419   |    0          |
| VGI.BK    |       1 |  0.371077   |    0          |
| CCET.BK   |       1 |  0.352399   |   -0          |
| BBL.BK    |       0 |  0.332323   |   -0          |
| BDMS.BK   |       0 |  0.327743   |   -0          |
| TTB.BK    |       0 |  0.320072   |   -0.00398402 |
| KKP.BK    |       1 |  0.166412   |   -0          |
| CPALL.BK  |       0 | -0.00258739 |   -0          |
| LH.BK     |       1 | -0.0576036  |    0          |
| CRC.BK    |       0 | -0.0796588  |   -0          |
| MINT.BK   |       1 | -0.271986   |    0          |
| CPN.BK    |       0 | -0.55546    |   -0          |
| RATCH.BK  |       1 | -0.585484   |   -0          |
| GPSC.BK   |       1 | -0.90325    |    0          |
| COM7.BK   |       1 | -0.931015   |   -0          |
| PTTGC.BK  |       1 | -0.933503   |    0          |
| MTC.BK    |       1 | -1.00614    |    0          |
| AWC.BK    |       0 | -1.03979    |   -0          |
| HMPRO.BK  |       1 | -1.44839    |   -0          |
| OSP.BK    |       1 | -1.45649    |   -0          |
| BEM.BK    |       0 | -1.469      |    0          |
| PTT.BK    |       1 | -1.47322    |    0          |
| BANPU.BK  |       1 | -1.64715    |    0          |
| TU.BK     |       0 | -1.68027    |    0          |
| AOT.BK    |       0 | -1.80048    |    0          |
| SCC.BK    |       1 | -1.80176    |    0          |
| EGCO.BK   |       1 | -2.1476     |   -0          |
| SCGP.BK   |       1 | -2.86439    |    0          |



### Walk-Forward Backtest (All Stocks)


| symbol    |   Sharpe_WF |   FinalEquity_WF |
|:----------|------------:|-----------------:|
| ADVANC.BK |    3.05198  |       0          |
| DELTA.BK  |    2.32155  |      -0          |
| WHA.BK    |    1.78242  |      -0          |
| TOP.BK    |    1.48833  |      -0          |
| TRUE.BK   |    1.27732  |       0          |
| KBANK.BK  |    1.1007   |       0          |
| TCAP.BK   |    0.957729 |      -0          |
| IVL.BK    |    0.929705 |      -0          |
| CBG.BK    |    0.907603 |      -0          |
| CPF.BK    |    0.822777 |      -0          |
| TISCO.BK  |    0.82072  |       0          |
| BCP.BK    |    0.729826 |       0          |
| KTB.BK    |    0.652395 |       0          |
| VGI.BK    |    0.630372 |      -0          |
| KTC.BK    |    0.524229 |      -0          |
| BJC.BK    |    0.481994 |      -0          |
| BH.BK     |    0.396525 |      -0          |
| BDMS.BK   |    0.327597 |       0          |
| TTB.BK    |    0.320072 |      -0.00398402 |
| BTS.BK    |    0.221813 |       0          |
| COM7.BK   |    0        |       1          |
| GPSC.BK   |    0        |       1          |
| HMPRO.BK  |    0        |       1          |
| OR.BK     |    0        |       1          |
| OSP.BK    |    0        |       1          |
| PTT.BK    |    0        |       1          |
| PTTGC.BK  |    0        |       1          |
| RATCH.BK  |    0        |       1          |
| SCGP.BK   |    0        |       1          |
| PTTEP.BK  |   -0.205802 |       0          |
| BBL.BK    |   -0.270539 |      -0          |
| CCET.BK   |   -0.528963 |       0          |
| CPN.BK    |   -0.546312 |      -0          |
| KKP.BK    |   -0.596449 |      -0          |
| BEM.BK    |   -0.640474 |      -0          |
| TU.BK     |   -0.644696 |       0          |
| MINT.BK   |   -0.901065 |      -0          |
| CRC.BK    |   -0.918722 |      -0          |
| MTC.BK    |   -1.22097  |       0          |
| AOT.BK    |   -1.2395   |       0          |
| CPALL.BK  |   -1.36062  |       0          |
| EGCO.BK   |   -1.61791  |      -0          |
| SCC.BK    |   -1.75602  |       0          |
| LH.BK     |   -1.82415  |      -0          |
| AWC.BK    |   -1.87893  |      -0          |
| BANPU.BK  |   -2.05497  |      -0          |