from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

RAW = Path("data/raw")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

SMA_WINDOWS = [10, 20, 50]
EMA_WINDOWS = [12, 26]
RSI_WINDOW = 14
STOCH_K, STOCH_D = 14, 3
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_NDEV = 20, 2

REQUIRED = ["Open","High","Low","Close","Volume"]

def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    # read raw, without assumptions
    df = pd.read_csv(csv_path)

    # find date column (or treat the first column as dates)
    date_col = None
    for cand in ["Date", "Datetime", "date", "DateTime"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]

    # parse dates and index
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "Date"

    # normalize column names commonly from yfinance
    rename_map = {c: c.title() for c in df.columns}  # open->Open, etc.
    if "Adj Close" in df.columns:
        rename_map["Adj Close"] = "AdjClose"
    df = df.rename(columns=rename_map)

    # coerce OHLCV to numeric
    for c in ["Open","High","Low","Close","AdjClose","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing core fields
    if set(REQUIRED).issubset(df.columns):
        df = df.dropna(subset=REQUIRED)
    else:
        missing = [c for c in REQUIRED if c not in df.columns]
        raise ValueError(f"Missing OHLCV columns: {missing}")

    # final sanity: ensure numeric dtypes
    for c in REQUIRED:
        if not is_numeric_dtype(df[c]):
            raise ValueError(f"Column {c} not numeric after coercion")

    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMA / EMA
    for w in SMA_WINDOWS:
        df[f"SMA_{w}"] = SMAIndicator(close=df["Close"], window=w).sma_indicator()
    for w in EMA_WINDOWS:
        df[f"EMA_{w}"] = EMAIndicator(close=df["Close"], window=w).ema_indicator()

    # RSI
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=RSI_WINDOW).rsi()

    # MACD
    macd = MACD(close=df["Close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=BB_WINDOW, window_dev=BB_NDEV)
    df["BB_low"]   = bb.bollinger_lband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_up"]    = bb.bollinger_hband()
    df["BB_width"] = (df["BB_up"] - df["BB_low"]) / df["BB_mid"]

    # Stochastic Oscillator
    st = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=STOCH_K, smooth_window=STOCH_D)
    df["STOCH_k"] = st.stoch()
    df["STOCH_d"] = st.stoch_signal()

    # Drop warmup NaNs
    df = df.dropna().copy()

    # Helpful returns (no label shift yet)
    df["ret_1d"] = df["Close"].pct_change().fillna(0)
    df["ret_5d"] = df["Close"].pct_change(5).fillna(0)
    return df

def process_all():
    rows = []
    csvs = sorted(RAW.glob("*.csv"))
    if not csvs:
        print("No raw CSVs found in data/raw")
        return
    for f in csvs:
        try:
            base = f.stem
            df = load_ohlcv(f)
            out = compute_indicators(df)
            out_file = OUT / f"{base}.processed.csv"
            out.to_csv(out_file)
            rows.append({
                "ticker": base,
                "rows_in": len(df), "rows_out": len(out),
                "first": out.index.min().date() if len(out) else None,
                "last":  out.index.max().date() if len(out) else None
            })
            print(f"[OK] {base}: {len(df)} -> {len(out)} rows")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")
            rows.append({"ticker": f.stem, "error": str(e)})

    summary = pd.DataFrame(rows)
    summary_path = REPORTS / "indicators_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved summary ->", summary_path)

if __name__ == "__main__":
    process_all()
