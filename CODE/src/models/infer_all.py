from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import pandas as pd

DATA = Path("data/windowed")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

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

def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for meta in sorted(DATA.glob("*_meta.json")):
        t = json.loads(meta.read_text())["ticker"]
        ckpt = REPORTS/f"bilstm_{t}.pt"
        if not ckpt.exists():
            rows.append({"ticker": t, "prob_up": None, "signal": None, "note": "no_model"})
            continue
        ck = torch.load(ckpt, map_location="cpu")
        in_dim = ck.get("in_dim", None)
        hidden = ck.get("hidden", 64)
        window = ck.get("window", 60)

        Xte = np.load(DATA/f"{t}_X_test.npy")
        if Xte.ndim != 3 or Xte.shape[1] != window or Xte.shape[2] != in_dim:
            rows.append({"ticker": t, "prob_up": None, "signal": None, "note": "shape_mismatch"})
            continue
        x = torch.tensor(Xte[-1:], dtype=torch.float32).to(device)

        model = BiLSTM(in_dim=in_dim, hidden=hidden).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        with torch.no_grad():
            prob_up = torch.sigmoid(model(x)).item()
        signal = "BUY" if prob_up >= 0.5 else "SELL/FLAT"
        rows.append({"ticker": t, "prob_up": round(prob_up,6), "signal": signal, "note": ""})

    pd.DataFrame(rows).to_csv(REPORTS/"infer_latest.csv", index=False)
    print(f"[DONE] saved -> {REPORTS/'infer_latest.csv'}")

if __name__ == "__main__":
    main()
