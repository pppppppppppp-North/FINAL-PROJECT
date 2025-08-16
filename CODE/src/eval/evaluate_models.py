#!/usr/bin/env python3
"""
Evaluate trained BiLSTM models on held-out data.

Handles both regression (float labels) and classification (0/1 labels).
Outputs per-symbol metrics to reports/<SYMBOL>/eval_metrics.json
and a global CSV summary at reports/eval_summary.csv.
"""

import os, json, math, argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = "DATA/processed"
CKPT_GLOB = "reports/*/model_*.pt"
DEVICE = "cpu"
BATCH_SIZE = 512

def load_tensor_pack(symbol):
    path = Path(DATA_DIR) / f"{symbol}.npz"
    pack = np.load(path, allow_pickle=True)
    return {k: pack[k] for k in pack.files}

def infer_symbol_from_ckpt(ckpt_path):
    p = Path(ckpt_path)
    if p.parent.name != "reports":
        return p.parent.name
    name = p.stem
    if "_" in name:
        return name.split("_")[0]
    return "UNKNOWN"

class BiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=2, dropout=0.2, task="regression"):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if task == "classification":
            self.out_act = nn.Sigmoid()

    def forward(self, x):
        y, _ = self.lstm(x)
        h = y[:, -1, :]
        out = self.head(h).squeeze(-1)
        if hasattr(self, "out_act"):
            out = self.out_act(out)
        return out

@torch.no_grad()
def predict(model, X):
    model.eval()
    preds = []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32, device=DEVICE)
        pb = model(xb).detach().cpu().numpy()
        preds.append(pb)
    return np.concatenate(preds)

def metrics_classification(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    out = {
        "accuracy": float((y_pred == y_true).mean().item()),
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
    y_true = y_true.astype(float); y_pred = y_pred.astype(float)
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))) * 100.0
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape_%": mape, "directional_acc": dir_acc}

def detect_task(y):
    u = np.unique(y)
    if u.size <= 3 and set(u.tolist()).issubset({0,1}):
        return "classification"
    return "regression"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_glob", default=CKPT_GLOB)
    ap.add_argument("--out_csv", default="reports/eval_summary.csv")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    for ckpt in sorted(glob.glob(args.ckpt_glob)):
        symbol = infer_symbol_from_ckpt(ckpt)
        try:
            pack = load_tensor_pack(symbol)
        except Exception as e:
            print(f"[WARN] Skip {symbol}: cannot load data ({e})")
            continue

        X_test, y_test = pack["X_test"], pack["y_test"]
        n_feat = X_test.shape[-1]
        task = detect_task(y_test)

        model = BiLSTM(n_feat=n_feat, hidden=64, num_layers=2, dropout=0.2, task=("classification" if task=="classification" else "regression"))
        state = torch.load(ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.to(DEVICE)

        y_pred = predict(model, X_test)

        if task == "classification":
            m = metrics_classification(y_test.astype(int), y_pred.astype(float), thresh=args.thresh)
        else:
            m = metrics_regression(y_test.astype(float), y_pred.astype(float))

        out_dir = Path("reports") / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval_metrics.json", "w") as f:
            json.dump({"symbol": symbol, "task": task, **m}, f, indent=2)

        print(symbol, task, m)
        rows.append({"symbol": symbol, "task": task, **m})

    if rows:
        df = pd.DataFrame(rows).sort_values("symbol")
        Path("reports").mkdir(exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] Wrote {args.out_csv}")
    else:
        print("[WARN] No rows written—check ckpt_glob and data files.")

if __name__ == "__main__":
    main()
