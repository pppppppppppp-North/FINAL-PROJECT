from pathlib import Path
import pandas as pd
import yfinance as yf

OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
TICKER = "PTT.BK"  # change later to loop your whole universe

def main():
    df = yf.download(TICKER, start="2014-01-01", auto_adjust=False)
    if df.empty:
        raise SystemExit(f"No data for {TICKER}")
    (OUT / f"{TICKER}.csv").write_text(df.to_csv())
    print("Saved:", OUT / f"{TICKER}.csv", "rows:", len(df))

if __name__ == "__main__":
    main()