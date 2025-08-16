import sys, json
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA = Path("data/windowed")
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CURVES = REPORTS / "curves"; CURVES.mkdir(parents=True, exist_ok=True)

class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class BiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)

def train_one_ticker(ticker, epochs=10, batch_size=64, lr=1e-3, hidden=64, dropout=0.2):
    Xtr = np.load(DATA/f"{ticker}_X_train.npy")
    ytr = np.load(DATA/f"{ticker}_y_train.npy")
    Xte = np.load(DATA/f"{ticker}_X_test.npy")
    yte = np.load(DATA/f"{ticker}_y_test.npy")

    train_dl = DataLoader(XYDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(XYDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM(in_dim=Xtr.shape[2], hidden=hidden, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = REPORTS / f"bilstm_{ticker}.pt"

    for epoch in range(1, epochs+1):
        model.train(); t_loss=0.0; t_acc=0.0; n=0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()*len(X)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            t_acc += (preds.eq(y)).float().sum().item()
            n += len(X)
        train_loss = t_loss/n; train_acc = t_acc/n

        model.eval(); v_loss=0.0; v_acc=0.0; n=0
        with torch.no_grad():
            for X, y in test_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                v_loss += loss.item()*len(X)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                v_acc += (preds.eq(y)).float().sum().item()
                n += len(X)
        val_loss = v_loss/n; val_acc = v_acc/n

        print(f"[{ticker}] epoch {epoch:02d} train_loss {train_loss:.4f} val_loss {val_loss:.4f} train_acc {train_acc:.3f} val_acc {val_acc:.3f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "in_dim": Xtr.shape[2],
                "hidden": hidden,
                "window": Xtr.shape[1],
                "task": "cls",
                "ticker": ticker
            }, ckpt_path)

    (REPORTS/f"bilstm_{ticker}_history.json").write_text(json.dumps(history, indent=2))
    import pandas as pd
    pd.DataFrame(history).to_csv(CURVES/f"{ticker}_history.csv", index=False)
    return best_val_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()
    train_one_ticker(args.ticker, args.epochs, args.batch_size, args.lr, args.hidden, args.dropout)

if __name__ == "__main__":
    main()
