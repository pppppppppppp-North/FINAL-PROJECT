#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd

PRED_CANDS = ["pred","y_pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]
DATE_CANDS = ["date","ds","timestamp","time"]
RET_CANDS  = ["ret","return","returns","y_ret","next_ret","r","y_next","future_ret","next_return","t+1_ret","ret_1d","ret_next"]
CLOSE_CANDS= ["close","Close","adj_close","Adj Close","Adj_Close","AdjClose","close_price","ClosePrice","Close*","Adj*"]

PRICE_DIRS = [
    Path("CODE/DATA"), Path("DATA"), Path("CODE/data"), Path("data"),
    Path("CODE/DATA/raw"), Path("CODE/DATA/processed"), Path("DATA/raw"), Path("DATA/processed")
]

def infer_symbol(path: Path):
    stem = path.stem
    m = re.match(r"([A-Za-z0-9\.\-]+)_test_preds", stem)
    return m.group(1) if m else stem

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def choose_pred_col(df):
    c = pick_col(df, PRED_CANDS)
    if c: return c
    nums = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    nums = [c for c in nums if c.lower() not in {"ret","return","returns"}]
    if not nums: return None
    stds = {c: float(pd.to_numeric(df[c], errors="coerce").std()) for c in nums}
    return max(stds, key=stds.get)

def load_price_for_symbol(sym: str):
    cands = []
    bases = [sym, sym.replace(".BK",""), sym.replace(".","_")]
    for d in PRICE_DIRS:
        for base in bases:
            cands += list(d.glob(f"**/{base}.csv"))
            cands += list(d.glob(f"**/{base}_prices.csv"))
            cands += list(d.glob(f"**/{base}*price*.csv"))
            cands += list(d.glob(f"**/{base}*ohlcv*.csv"))
    # also any csv containing symbol
    for d in PRICE_DIRS:
        cands += list(d.glob(f"**/*{bases[0]}*.csv"))
    seen = []
    for p in cands:
        if p.exists() and p.is_file() and str(p) not in seen:
            seen.append(str(p))
            try:
                df = pd.read_csv(p)
                return p, df
            except Exception:
                continue
    return None, None

def derive_returns_from_df(df):
    rcol = pick_col(df, RET_CANDS)
    if rcol:
        r = pd.to_numeric(df[rcol], errors="coerce")
        return r, rcol, "ret_col"
    ycol = None
    for c in ["y_true","target","label","y","direction"]:
        if c in df.columns:
            ycol = c; break
    if ycol is not None:
        y = pd.to_numeric(df[ycol], errors="coerce")
        if set(y.dropna().unique().astype(int)).issubset({0,1}):
            r = np.where(y.astype(int)>0, 1.0, -1.0)
            return pd.Series(r), ycol, "y_binary_pm1"
        else:
            return y, ycol, "y_float"
    ccol = None
    for c in CLOSE_CANDS:
        if c in df.columns:
            ccol = c; break
    if ccol:
        close = pd.to_numeric(df[ccol], errors="coerce")
        r = close.pct_change().shift(-1)
        return r, ccol, "pct_from_close_fwd"
    return None, None, "none"

def align_by_date_or_tail(pred_df, price_df, pred_date, price_date, price_ret):
    if pred_date and price_date and pred_date in pred_df.columns and price_date in price_df.columns:
        a = pred_df.rename(columns={pred_date:"date"})
        b = price_df.rename(columns={price_date:"date"})
        a["date"] = pd.to_datetime(a["date"])
        b["date"] = pd.to_datetime(b["date"])
        m = a.merge(b[["date", price_ret]], on="date", how="left")
        return m[["date"]], m[price_ret]
    # fall back to tail align
    n = len(pred_df)
    pr = price_df[price_ret].dropna().reset_index(drop=True)
    if len(pr) >= n:
        pr = pr.iloc[-n:].reset_index(drop=True)
        if pred_date and pred_date in pred_df.columns:
            return pred_df[[pred_date]].rename(columns={pred_date:"date"}), pr
        else:
            return None, pr
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--outdir", default="CODE/reports/preds_ready")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(".").glob(args.glob))
    made=0; skipped=0
    for f in files:
        try:
            dfp = pd.read_csv(f)
        except Exception:
            skipped+=1; continue
        pred_col = choose_pred_col(dfp)
        if pred_col is None:
            skipped+=1; continue
        date_col = pick_col(dfp, DATE_CANDS)
        r_ser, r_src, r_how = derive_returns_from_df(dfp)
        if r_ser is None:
            sym = infer_symbol(f)
            p_path, dprice = load_price_for_symbol(sym)
            if dprice is not None:
                r_ser2, r_src2, r_how2 = derive_returns_from_df(dprice)
                if r_ser2 is not None:
                    pdate = pick_col(dprice, DATE_CANDS)
                    d_aligned, r_aligned = align_by_date_or_tail(dfp, dprice, date_col, pdate, r_src2)
                    if r_aligned is not None:
                        r_ser = r_aligned
                        if d_aligned is not None:
                            dfp = dfp.join(d_aligned.reset_index(drop=True))
                            date_col = "date"
        if r_ser is None:
            skipped+=1; continue
        df_out = pd.DataFrame({"pred": pd.to_numeric(dfp[pred_col], errors="coerce")})
        df_out["ret"] = pd.to_numeric(r_ser, errors="coerce")
        if date_col and date_col in dfp.columns:
            df_out["date"] = dfp[date_col]
        df_out = df_out.dropna(subset=["pred","ret"]).reset_index(drop=True)
        sym = infer_symbol(f)
        out = outdir / f"{sym}_preds_ready.csv"
        df_out.to_csv(out, index=False)
        made+=1
    print(f"PREPARED={made} SKIPPED={skipped} OUTDIR={outdir}")
if __name__ == "__main__":
    main()
