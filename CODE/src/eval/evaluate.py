from pathlib import Path
import argparse, json
import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA = Path("data/windowed")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
PREDS = REPORTS/"preds"; PREDS.mkdir(parents=True, exist_ok=True)

class BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True, bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*d, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc(out)
        return out

def eval_one(ticker):
    ckpt = REPORTS/f"bilstm_{ticker}.pt"
    if not ckpt.exists():
        return {"ticker": ticker, "note": "no_model"}
    ck = torch.load(ckpt, map_location="cpu")
    in_dim = ck.get("in_dim"); hidden = ck.get("hidden", 64); window = ck.get("window", 60)

    Xte = np.load(DATA/f"{ticker}_X_test.npy"); yte = np.load(DATA/f"{ticker}_y_test.npy")
    if Xte.ndim != 3 or Xte.shape[1] != window or Xte.shape[2] != in_dim:
        return {"ticker": ticker, "note": "shape_mismatch"}

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    with torch.no_grad():
        logits = []
        for i in range(Xte.shape[0]):
            x = torch.tensor(Xte[i:i+1], dtype=torch.float32).to(device)
            logit = model(x).cpu().numpy().ravel()[0]
            logits.append(logit)
    logits = np.array(logits)
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    y = yte.astype(int).ravel()
    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    try:
        auc  = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")

    df = pd.DataFrame({"y": y, "prob_up": probs, "pred": preds})
    df.to_csv(PREDS/f"{ticker}_test_preds.csv", index=False)

    return {"ticker": ticker, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "note": ""}

def list_tickers():
    metas = sorted(DATA.glob("*_meta.json"))
    return [json.loads(p.read_text())["ticker"] for p in metas]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="ALL")
    args = ap.parse_args()
    tickers = list_tickers() if args.ticker.upper()=="ALL" else [args.ticker]
    rows = [eval_one(t) for t in tickers]
    pd.DataFrame(rows).to_csv(REPORTS/"eval_summary.csv", index=False)
    print(REPORTS/"eval_summary.csv")
if __name__ == "__main__":
    main()
