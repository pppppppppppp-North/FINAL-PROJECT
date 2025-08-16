import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

DATA = Path("data/windowed")
REPORTS = Path("reports")

def die(msg):
    print(msg); sys.exit(1)

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
    if len(sys.argv) < 2:
        die("Usage: python src/models/infer_bilstm.py <TICKER>  (e.g., PTT.BK)")
    ticker = sys.argv[1]

    ckpt_path = REPORTS / f"bilstm_{ticker}.pt"
    if not ckpt_path.exists():
        die(f"[ERR] Model not found: {ckpt_path} (train first with bilstm_baseline.py)")

    # Load checkpoint (compatible with older torch)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim  = ckpt.get("in_dim")
    window  = ckpt.get("window", 60)
    task    = ckpt.get("task", "cls")
    hidden  = ckpt.get("hidden", 64)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Ticker={ticker}  Device={device}  Task={task}")

    model = BiLSTM(in_dim=in_dim, hidden=hidden, dropout=0.2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[2/4] Loaded checkpoint -> {ckpt_path.name}  (window={window}, in_dim={in_dim})")

    Xte_path = DATA / f"{ticker}_X_test.npy"
    if not Xte_path.exists():
        die(f"[ERR] Not found: {Xte_path} (run window.py)")
    Xte = np.load(Xte_path)
    if Xte.ndim != 3 or Xte.shape[1] != window or Xte.shape[2] != in_dim:
        die(f"[ERR] Shape mismatch: X_test={Xte.shape}, expected (*,{window},{in_dim})")

    x = torch.tensor(Xte[-1:], dtype=torch.float32).to(device)  # most recent window
    print(f"[3/4] Latest window shape: {tuple(x.shape)} (B,T,F)")

    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
    dt = (time.time() - t0)*1000.0
    print(f"[4/4] Inference done in {dt:.1f} ms")

    if task == "cls":
        prob_up = torch.sigmoid(logits).item()
        signal = "BUY (UP)" if prob_up >= 0.5 else "SELL/FLAT (DOWN)"
        print("\n===== RESULT =====")
        print(f"Ticker: {ticker}")
        print(f"Probability UP (next horizon): {prob_up:.4f}")
        print(f"Signal: {signal}")

        # optional CSV log
        out = REPORTS / "infer_latest.csv"
        line = f"{ticker},{prob_up:.6f},{signal}\n"
        if not out.exists(): out.write_text("ticker,prob_up,signal\n")
        with out.open("a") as f: f.write(line)
        print(f"\nSaved → {out}")
    else:
        pred = logits.item()
        print("\n===== RESULT =====")
        print(f"Ticker: {ticker}")
        print(f"Predicted next-horizon return: {pred:.5f}")

if __name__ == "__main__":
    main()
