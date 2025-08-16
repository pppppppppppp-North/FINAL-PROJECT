#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

CAND_TRUE = ["y_true","true","label","target","y","actual","direction"]
CAND_PRED = ["y_pred","pred","prediction","prob","proba","score","yhat","pred_prob","pred_score"]
CAND_RET  = ["ret","return","returns","y_ret","next_ret","r","y_next"]
CAND_DATE = ["date","timestamp","time","ds"]

def infer_symbol(path):
    stem = Path(path).stem
    m = re.match(r"([A-Za-z0-9\.\-_]+)_test_preds", stem)
    return m.group(1) if m else stem

def detect_cols(df):
    yt=yp=rt=dt=None
    for c in CAND_TRUE:
        if c in df.columns: yt=c; break
    for c in CAND_PRED:
        if c in df.columns: yp=c; break
    for c in CAND_RET:
        if c in df.columns: rt=c; break
    for c in CAND_DATE:
        if c in df.columns: dt=c; break
    if yp is None:
        num_cols=[c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if num_cols: yp=num_cols[0]
    return yt,yp,rt,dt

def detect_task(y):
    if y is None: return "regression"
    u = pd.Series(y).dropna().unique()
    try:
        s=set(pd.Series(u).astype(int).tolist())
        if len(s)<=3 and s.issubset({0,1}): return "classification"
    except Exception:
        pass
    return "regression"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="CODE/reports/preds/*_test_preds.csv")
    ap.add_argument("--outdir", default="CODE/reports/diagnostics")
    ap.add_argument("--thr_map", default="")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    thr_map = {}
    if args.thr_map and Path(args.thr_map).exists():
        thr_map = pd.read_csv(args.thr_map).set_index("symbol")["thr"].to_dict()

    files = sorted([str(p) for p in Path(".").glob(args.glob)])
    rows = []
    for f in files:
        df = pd.read_csv(f)
        yt, yp, rt, dt = detect_cols(df)
        if yp is None: continue
        sym = infer_symbol(f)
        y_pred = df[yp].astype(float).values
        y_true = df[yt].values if yt is not None else None
        task = detect_task(y_true)

        outdir = Path(args.outdir)/sym
        outdir.mkdir(parents=True, exist_ok=True)

        if task == "classification":
            y = pd.Series(y_true).astype(int).values
            fpr, tpr, _ = roc_curve(y, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.title(f"ROC: {sym}")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
            plt.tight_layout()
            plt.savefig(outdir/"roc.png", dpi=150); plt.close()

            prec, rec, _ = precision_recall_curve(y, y_pred)
            ap = average_precision_score(y, y_pred)
            plt.figure()
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            plt.title(f"PR Curve: {sym}")
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
            plt.tight_layout()
            plt.savefig(outdir/"pr.png", dpi=150); plt.close()

            thr = float(thr_map.get(sym, 0.5))
            y_hat = (y_pred >= thr).astype(int)
            cm = confusion_matrix(y, y_hat, labels=[0,1])
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i,j], ha="center", va="center")
            plt.xticks([0,1], ["Pred 0","Pred 1"])
            plt.yticks([0,1], ["True 0","True 1"])
            plt.title(f"Confusion (thr={thr:.2f}): {sym}")
            plt.tight_layout()
            plt.savefig(outdir/"confusion.png", dpi=150); plt.close()

            bins = np.linspace(0,1,11)
            dfb = pd.DataFrame({"p":y_pred, "y":y})
            dfb["bin"] = pd.cut(dfb["p"], bins, include_lowest=True)
            calib = dfb.groupby("bin").agg(p_mean=("p","mean"), y_rate=("y","mean"), n=("y","size")).reset_index(drop=True)
            plt.figure()
            plt.plot([0,1],[0,1], linestyle="--")
            plt.scatter(calib["p_mean"], calib["y_rate"])
            plt.title(f"Calibration: {sym}")
            plt.xlabel("Predicted prob"); plt.ylabel("Observed freq")
            plt.tight_layout()
            plt.savefig(outdir/"calibration.png", dpi=150); plt.close()

        else:
            plt.figure()
            if yt is not None:
                plt.scatter(df[yt].astype(float), y_pred, s=8)
                plt.xlabel("y_true")
                plt.ylabel("y_pred")
                plt.title(f"True vs Pred: {sym}")
                plt.tight_layout()
                plt.savefig(outdir/"true_vs_pred.png", dpi=150); plt.close()

                resid = y_pred - df[yt].astype(float).values
                plt.figure()
                plt.hist(resid, bins=40)
                plt.title(f"Residuals: {sym}")
                plt.tight_layout()
                plt.savefig(outdir/"residual_hist.png", dpi=150); plt.close()

            plt.figure()
            plt.hist(y_pred, bins=40)
            plt.title(f"Pred Distribution: {sym}")
            plt.tight_layout()
            plt.savefig(outdir/"pred_hist.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
