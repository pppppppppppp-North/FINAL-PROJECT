#!/usr/bin/env python3
import argparse, re, math, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

CAND_TRUE = ["y_true","true","label","target","y","actual"]
CAND_PRED = ["y_pred","pred","prediction","prob","proba","score","yhat"]

def detect_cols(df):
    y_true = None; y_pred = None; date = None
    for c in CAND_TRUE:
        if c in df.columns: y_true = c; break
    for c in CAND_PRED:
        if c in df.columns: y_pred = c; break
    for c in ["date","timestamp","time","ds"]:
        if c in df.columns: date = c; break
    if y_true is None or y_pred is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) >= 2:
            y_true, y_pred = num_cols[0], num_cols[1]
        elif len(num_cols) == 1:
            y_pred = num_cols[0]
            other = [c for c in df.columns if c != y_pred][0]
            y_true = other
        else:
            raise ValueError("Could not infer columns")
    return y_true, y_pred, date

def detect_task(y):
    u = pd.Series(y).dropna().unique()
    if len(u) <= 3 and set(pd.Series(u).astype(int).tolist()).issubset({0,1}):
        return "classification"
    return "regression"

def metrics_classification(y_true, y_prob, thr=0.5):
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = float("nan")
    return out

def metrics_regression(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))) * 100.0
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape_%": mape, "directional_acc": dir_acc}

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="reports/*_test_preds.csv")
    ap.add_argument("--out_csv", default="reports/eval_from_preds_summary.csv")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        try:
            yt_col, yp_col, _ = detect_cols(df)
        except Exception:
            continue
        sym = infer_symbol(f)
        task = detect_task(df[yt_col].values)
        if task == "classification":
            m = metrics_classification(df[yt_col].values, df[yp_col].values, thr=args.thresh)
        else:
            m = metrics_regression(df[yt_col].values, df[yp_col].values)
        outdir = Path("reports")/sym
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir/"eval_metrics_from_preds.json","w") as fh:
            json.dump({"symbol": sym, "task": task, **m}, fh, indent=2)
        mrow = {"symbol": sym, "task": task, **m}
        rows.append(mrow)
        print(sym, mrow)
    if rows:
        pd.DataFrame(rows).sort_values("symbol").to_csv(args.out_csv, index=False)
        print(f"[OK] Wrote {args.out_csv}")
    else:
        print("[WARN] No prediction CSVs found or columns not detected")

if __name__ == "__main__":
    main()
