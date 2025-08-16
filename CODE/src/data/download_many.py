from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import date

UNIVERSE = Path("data/tickers_set50_2025H2.txt")
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

def fetch(ticker, start="2014-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.to_csv(RAW_DIR / f"{ticker}.csv")
    return df

def main():
    tickers = [t.strip() for t in UNIVERSE.read_text().splitlines() if t.strip()]
    rows = []
    for t in tickers:
        try:
            print(f"Downloading {t} ...", end=" ")
            df = fetch(t)
            first, last = df.index.min().date(), df.index.max().date()
            na_ratio = float(df.isna().sum().sum()) / float(df.size)
            print(f"ok ({len(df)} rows, {first} → {last})")
            rows.append({"ticker": t, "rows": len(df), "first": first, "last": last, "na_ratio": round(na_ratio, 4)})
        except Exception as e:
            print("FAILED:", e)
            rows.append({"ticker": t, "rows": 0, "first": None, "last": None, "na_ratio": None, "error": str(e)})

    summary = pd.DataFrame(rows)
    out = REPORTS / f"ohlcv_summary_{date.today().isoformat()}.csv"
    summary.to_csv(out, index=False)
    print("\nSaved summary ->", out)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
