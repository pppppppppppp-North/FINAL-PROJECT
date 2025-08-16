from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
OUT = Path("data/windowed"); OUT.mkdir(parents=True, exist_ok=True)

# config — tweak later
WINDOW = 60          # lookback length (days)
HORIZON = 5          # predict 5-day ahead return
SPLIT_DATE = "2022-01-01"
TASK = "cls"         # "cls" = classification, "reg" = regression

FEATURES = [
    "Close","Volume",
    "SMA_10","SMA_20","EMA_12","EMA_26",
    "RSI_14","MACD","MACD_signal","MACD_hist",
    "BB_width","STOCH_k","STOCH_d",
    "ret_1d","ret_5d"
]

def load_one(path: Path) -> pd.DataFrame:
    print(f"  [1/5] Loading {path.name}")
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    cols = [c for c in FEATURES if c in df.columns]
    if "Close" not in cols:
        raise ValueError("Close not found in processed file.")
    return df[cols].dropna()

def future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0

def make_windows(arrX: np.ndarray, arrY: np.ndarray, window: int):
    Xs, Ys = [], []
    for t in range(len(arrY) - window + 1):
        end = t + window
        y = arrY[end - 1]
        Xs.append(arrX[t:end])
        Ys.append(y)
    return np.asarray(Xs), np.asarray(Ys)

def process_ticker(csv_path: Path):
    ticker = csv_path.stem.replace(".processed","")
    try:
        df = load_one(csv_path)
        print(f"  [2/5] Splitting train/test at {SPLIT_DATE}")
        df_train = df.loc[:pd.Timestamp(SPLIT_DATE) - pd.Timedelta(days=1)].copy()
        df_test  = df.loc[pd.Timestamp(SPLIT_DATE):].copy()

        if len(df_train) < WINDOW + 100 or len(df_test) < WINDOW + 20:
            print(f"  [SKIP] {ticker}: not enough data")
            return

        print(f"  [3/5] Creating labels (HORIZON={HORIZON}, TASK={TASK})")
        y_train_raw = future_return(df_train["Close"], HORIZON)
        y_test_raw  = future_return(df_test["Close"],  HORIZON)

        if TASK == "cls":
            y_train = (y_train_raw > 0).astype(np.int8)
            y_test  = (y_test_raw  > 0).astype(np.int8)
        else:
            y_train = y_train_raw.astype(np.float32)
            y_test  = y_test_raw.astype(np.float32)

        df_train = df_train.iloc[:-HORIZON]
        df_test  = df_test.iloc[:-HORIZON]
        y_train  = y_train.iloc[:-HORIZON]
        y_test   = y_test.iloc[:-HORIZON]

        print("  [4/5] Scaling features (fit on train only)")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train.values)
        X_test  = scaler.transform(df_test.values)

        print("  [5/5] Building windows")
        Xtr, ytr = make_windows(X_train, y_train.values, WINDOW)
        Xte, yte = make_windows(X_test,  y_test.values,  WINDOW)

        # save arrays & meta
        base = OUT / ticker
        np.save(str(base) + "_X_train.npy", Xtr)
        np.save(str(base) + "_y_train.npy", ytr)
        np.save(str(base) + "_X_test.npy",  Xte)
        np.save(str(base) + "_y_test.npy",  yte)

        meta = {
            "ticker": ticker,
            "features_used": [c for c in FEATURES if c in df.columns],
            "window": WINDOW, "horizon": HORIZON, "task": TASK,
            "split_date": SPLIT_DATE,
            "train_rows": int(len(df_train)), "test_rows": int(len(df_test)),
            "X_train_shape": list(Xtr.shape), "X_test_shape": list(Xte.shape)
        }
        (OUT / f"{ticker}_meta.json").write_text(json.dumps(meta, indent=2))
        np.save(str(base) + "_scaler_mean.npy", scaler.mean_)
        np.save(str(base) + "_scaler_scale.npy", scaler.scale_)
        print(f"[OK] {ticker}: train{Xtr.shape} test{Xte.shape}\n")
    except Exception as e:
        print(f"[FAIL] {ticker}: {e}\n")

def main():
    files = sorted(PROCESSED.glob("*.processed.csv"))
    if not files:
        print("No processed files found.")
        return
    for f in files:
        process_ticker(f)

if __name__ == "__main__":
    main()
