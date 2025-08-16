from pathlib import Path
import json
import argparse
import pandas as pd
from bilstm_baseline import train_one_ticker

DATA = Path("data/windowed")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

def list_tickers():
    metas = sorted(DATA.glob("*_meta.json"))
    tickers = [json.loads(p.read_text())["ticker"] for p in metas]
    return tickers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    rows = []
    for t in list_tickers():
        try:
            print(f"[RUN] {t}")
            best = train_one_ticker(t, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, dropout=args.dropout)
            rows.append({"ticker": t, "best_val_acc": best})
        except Exception as e:
            print(f"[FAIL] {t}: {e}")
            rows.append({"ticker": t, "best_val_acc": None, "error": str(e)})

    pd.DataFrame(rows).to_csv(REPORTS/"models_summary.csv", index=False)
    print(f"[DONE] saved -> {REPORTS/'models_summary.csv'}")

if __name__ == "__main__":
    main()
