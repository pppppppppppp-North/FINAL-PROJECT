# FINAL PROJECT — Evaluation & Backtest Summary

## Top 10 Symbols by Sharpe

| symbol    | task           |   thr |    CAGR |   Sharpe |   MaxDD |    WinRate |   FinalEquity |
|:----------|:---------------|------:|--------:|---------:|--------:|-----------:|--------------:|
| ADVANC.BK | classification |   0.5 | -1      |  2.36791 |  -1     | 0.192593   |         -0    |
| DELTA.BK  | classification |   0.5 | -1      |  2.11989 | nan     | 0.111111   |          0    |
| BH.BK     | classification |   0.5 | -1      |  1.99314 |  -1     | 0.233333   |          0    |
| WHA.BK    | classification |   0.5 | -1      |  1.82356 |  -1     | 0.176543   |          0    |
| KBANK.BK  | classification |   0.5 | -1      |  1.79726 |  -1     | 0.201235   |          0    |
| TRUE.BK   | classification |   0.5 | -1      |  1.64249 |  -1     | 0.120988   |         -0    |
| TOP.BK    | classification |   0.5 | -1      |  1.63391 |  -1.001 | 0.244444   |         -0    |
| OR.BK     | classification |   0.5 |  4.6056 |  1.58389 |  -0.001 | 0.00987654 |        254.85 |
| IVL.BK    | classification |   0.5 | -1      |  1.45842 |  -1     | 0.112346   |         -0    |
| CBG.BK    | classification |   0.5 | -1      |  1.41269 |  -1.001 | 0.241975   |         -0    |

## Portfolio Metrics (equal-weight)

- Sharpe: -0.502
- CAGR: -99.70%
- MaxDD: -100.00%
- WinRate: 43.70%
- Final Equity: 0.00


![Portfolio Equity](portfolio/portfolio_equity.png)

![Rolling Sharpe](portfolio/rolling_sharpe.png)

![Rolling Drawdown](portfolio/rolling_drawdown.png)

## Per-Symbol Ranking (Top 20)

| symbol    |   Sharpe |    CAGR |   FinalEquity |
|:----------|---------:|--------:|--------------:|
| ADVANC.BK | 2.36791  | -1      |         -0    |
| DELTA.BK  | 2.11989  | -1      |          0    |
| BH.BK     | 1.99314  | -1      |          0    |
| WHA.BK    | 1.82356  | -1      |          0    |
| KBANK.BK  | 1.79726  | -1      |          0    |
| TRUE.BK   | 1.64249  | -1      |         -0    |
| TOP.BK    | 1.63391  | -1      |         -0    |
| OR.BK     | 1.58389  |  4.6056 |        254.85 |
| IVL.BK    | 1.45842  | -1      |         -0    |
| CBG.BK    | 1.41269  | -1      |         -0    |
| BCP.BK    | 1.32859  | -1      |          0    |
| CPF.BK    | 1.157    | -1      |         -0    |
| TCAP.BK   | 0.991513 | -1      |         -0    |
| TISCO.BK  | 0.946146 | -1      |          0    |
| KTB.BK    | 0.870558 | -1      |          0    |
| BTS.BK    | 0.673123 | -1      |          0    |
| PTTEP.BK  | 0.629878 | -1      |          0    |
| BJC.BK    | 0.547047 | -1      |          0    |
| KTC.BK    | 0.524419 | -1      |          0    |
| VGI.BK    | 0.371077 | -1      |          0    |

## Transaction Cost Sensitivity

|   tc_bps |   mean_sharpe |   mean_cagr |
|---------:|--------------:|------------:|
|        0 |     0.0721035 |   -0.877968 |
|        5 |     0.0710106 |   -0.875344 |
|       10 |     0.0699177 |   -0.875431 |
|       15 |     0.0688247 |   -0.875518 |
|       20 |     0.0677317 |   -0.875606 |
|       30 |     0.0655455 |   -0.87578  |
|       50 |     0.0611727 |   -0.876129 |
