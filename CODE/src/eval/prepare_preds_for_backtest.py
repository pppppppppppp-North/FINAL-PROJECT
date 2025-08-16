#!/usr/bin/env python3
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd

PRED_CANDS = ["pred","y_pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]
DATE_CANDS = ["date","ds","timestamp","time"]
RET_CANDS  = ["ret","return","returns","y_ret","next_ret","r","y_next"]
CLOSE_CANDS= ["close","Close","adj_close","Adj Close","Adj_Close","AdjClose","close_price","ClosePrice"]

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def pick_pred(df):
    for c in PRED_CANDS:
        if c in df.columns: return c
    num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    num = [c for c in num if c.lower() not in {"ret","return","returns"}]
    if not num: return None
    stds = {c: float(df[c].astype(float).std()) for c in num}
    return max(stds, key=stds.get)

def pick_date(df):
    for c in DATE_CANDS:
        if c in df.columns: return c
    return None

def has_binary_y(df):
    if "y_true" not in df.columns: return False
    try:
        u = set(df["y_true"].dropna().astype(int).unique().tolist())
        return u.issubset({0,1})
    except Exception:
        return False

def derive_ret(df):
    for c in RET_CANDS:
        if c in df.columns:
            return df[c].astype(float).values, c, "from_ret_col"
    if "y_true" in df.columns:
        y = df["y_true"]
        try:
            y_float = y.astype(float)
            if not has_binary_y(df):
                return y_float.values, "y_true", "from_y_true_float"
            else:
                return np.where(y.astype(int)>0, 1.0, -1.0), "y_true", "from_y_true_binary_pm1"
        except Exception:
            pass
    for c in CLOSE_CANDS:
        if c in df.columns:
            close = pd.to_numeric(df[c], errors="coerce")
            r = close.pct_change().shift(-1)
            return r.values, c, "from_close_pct_fwd"
    return None, None, "unavailable"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--outdir", default="CODE/reports/preds_ready")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(".").glob(args.glob))
    made = 0; skipped = 0
    for f in files:
        df = pd.read_csv(f)
        pcol = pick_pred(df)
        if pcol is None:
            skipped += 1
            continue
        dcol = pick_date(df)
        rvals, rsrc, why = derive_ret(df)
        if rvals is None:
            skipped += 1
            continue
        ren = {}
        ren[pcol] = "pred"
        if dcol: ren[dcol] = "date"
        df = df.rename(columns=ren)
        df["ret"] = rvals
        keep = ["pred","ret"] + (["date"] if "date" in df.columns else [])
        df = df[keep].dropna(subset=["pred","ret"]).reset_index(drop=True)
        sym = infer_symbol(f)
        out = outdir / f"{sym}_test_preds_ready.csv"
        df.to_csv(out, index=False)
        made += 1
    print(f"PREPARED={made} SKIPPED={skipped} OUTDIR={outdir}")
if __name__ == "__main__":
    main()
